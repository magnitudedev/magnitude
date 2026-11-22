use super::super::*;
use super::*;
use crate::ReadoutHead;

impl<'a> QualificationView<'a> {
    pub(super) fn qualify_head(&self, charges: &seismic::Charges) -> Result<(), CatalogFailure> {
        let (Some(head_plan), Some(AttestedDrafter::Head(head))) =
            (self.plan.head(), self.programs.drafter.as_ref())
        else {
            return Ok(());
        };
        let (hidden, vocabulary) = (self.geometry.hidden, self.geometry.vocabulary);
        for (index, (&binding, block)) in head_plan.blocks().iter().zip(&head.blocks).enumerate() {
            let label = format!("{binding:?}");
            // The head block's feed-forward sublayer.
            let scope = magnitude_family_contracts::WeightScope::HeadSublayer(
                magnitude_family_contracts::SublayerIndex {
                    block: u32::try_from(index)
                        .map_err(|error| qualification("head", "fixture", error))?,
                    sublayer: 1,
                },
            );
            // One (token, status) selection row.
            let tokens = semantic_zeros(charges, Element::i32(), &[1, 2], "head", &label)?;
            let direct = semantic_zeros(charges, Element::i32(), &[0, 1], "head", &label)?;
            let table = semantic_pattern(
                charges,
                binding.embedding_table,
                &[1, hidden],
                "head",
                &label,
            )?;
            let streams = binding.streams.map_or(1, |s| s.geometry.streams);
            let form = SublayerRows { form: crate::operators::InputForm { parts: streams, normalized: binding.streams.is_none() }, streams, width: binding.streams.map_or(0, |s| s.geometry.streams) };
            let pattern = (0..hidden * streams)
                .map(|column| [1.0_f32, -2.0, 3.0, -4.0][column as usize % 4])
                .collect::<Vec<_>>();
            let conditioning = semantic_dense_values(
                charges,
                binding.conditioning,
                &[1, hidden * streams],
                &pattern,
                "head",
                &label,
            )?;
            let embedding_norm =
                semantic_ones(charges, binding.embedding_norm, &[hidden], "head", &label)?;
            let hidden_norm =
                semantic_ones(charges, binding.hidden_norm, &[hidden * streams], "head", &label)?;
            let combine = semantic_pattern(
                charges,
                binding.combine,
                &[hidden, 2 * hidden],
                "head",
                &label,
            )?;
            let features = block
                .input
                .first()
                .call(charges, draft_rows::Args {
                    tokens: &tokens,
                    directory: &direct,
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
                charges,
                &block.attention,
                binding.attention,
                form,
                &label,
            )?;
            // The attention block keeps its residual (zero weights), so the
            // feed-forward stage continues from the head input.
            let attended = features;
            let out_rows = semantic_zeros(charges, Element::i32(), &[1], "head", &label)?;
            let advanced = match (binding.feed_forward, &block.feed_forward) {
                (FeedForwardProgramSlot::Dense(binding), AttestedFeedForward::Dense(kernels)) => {
                    let intermediate = self.weight_shape(
                        scope,
                        magnitude_family_contracts::WeightKind::DenseGate,
                        "dense_expand",
                    )?[0];
                    let norm = semantic_zeros(charges, binding.norm, &[hidden], "head", &label)?;
                    let gate = semantic_zeros(
                        charges,
                        binding.gate,
                        &[intermediate, hidden],
                        "head",
                        &label,
                    )?;
                    let up = semantic_zeros(
                        charges,
                        binding.up,
                        &[intermediate, hidden],
                        "head",
                        &label,
                    )?;
                    let down = semantic_zeros(
                        charges,
                        binding.down,
                        &[hidden, intermediate],
                        "head",
                        &label,
                    )?;
                    let product = kernels
                        .expand
                        .first()
                        .call(charges, dense_expand::Args {
                            residual: &attended,
                            norm: &norm,
                            gate_weight: &gate,
                            up_weight: &up,
                            out_rows: &out_rows,
                            eps: 1.0e-5,
                            activation: 0,
                            gate_scale: &unit_scale(
                                charges,
                                kernels.expand.first(),
                                "GS",
                                "dense_expand",
                                &label,
                            )?,
                            up_scale: &unit_scale(
                                charges,
                                kernels.expand.first(),
                                "US",
                                "dense_expand",
                                &label,
                            )?,
                            gate_bound: &any_row_bound(
                                charges,
                                kernels.expand.first(),
                                "GB",
                                "dense_expand",
                                &label,
                            )?,
                            up_bound: &any_row_bound(
                                charges,
                                kernels.expand.first(),
                                "UB",
                                "dense_expand",
                                &label,
                            )?,
                            glu_limit: &glu_limit(charges, binding.limit, "dense_expand", &label)?,
                        })
                        .map_err(|error| qualification_dynamic("dense_expand", &label, error))?
                        .value;
                    let SublayerOutput::Residual(output) = &kernels.output else {
                        return Err(qualification_dynamic(
                            "dense_output",
                            &label,
                            "draft head feed-forward has a post-norm tail",
                        ));
                    };
                    let output = output.first();
                    output
                        .call(charges, dense_output::Args {
                            residual: &attended,
                            product: &product,
                            down_weight: &down,
                            out_rows: &out_rows,
                            down_scale: &unit_scale(charges, output, "DS", "dense_output", &label)?,
                            down_bound: &any_row_bound(
                                charges,
                                output,
                                "DB",
                                "dense_output",
                                &label,
                            )?,
                        })
                        .map_err(|error| qualification_dynamic("dense_output", &label, error))?
                        .value
                }
                (FeedForwardProgramSlot::Routed(binding), AttestedFeedForward::Routed(kernels)) => {
                    let reads = super::target::ExpertReads::routed(charges, binding)?;
                    super::target::qualify_routed(charges, binding, kernels, form, &reads)?;
                    attended
                }
                (
                    FeedForwardProgramSlot::GeneralRouted(binding),
                    AttestedFeedForward::GeneralRouted(kernels),
                ) => {
                    let reads = super::target::ExpertReads::general(charges, binding)?;
                    super::target::qualify_general_routed(charges, binding, kernels, &reads)?;
                    attended
                }
                _ => {
                    return Err(qualification_dynamic(
                        "head",
                        &label,
                        "feed-forward binding and attestation disagree",
                    ));
                }
            };
            require_finite_nonzero_f32(&advanced, "head_feed_forward", &label)?;
            let output_norm =
                semantic_ones(charges, binding.output_norm, &[if binding.streams.is_some() { 0 } else { hidden }], "head", &label)?;
            let parts = readout_parts(
                &advanced,
                block.features.specialization().statics(),
                "readout_features_rows",
                &label,
            )?;
            let projected_features = block
                .features
                .call(charges, readout_features_rows::Args {
                    hidden: &parts,
                    norm: &output_norm,
                    out_rows: &out_rows,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("readout_features_rows", &label, error))?
                .value;
            let logits = match &block.logits {
                ProjectionKernels::Qualified {
                    packed,
                    progressive,
                    ..
                } => {
                    super::target::qualify_progressive(
                        charges,
                        progressive,
                        &advanced,
                        &output_norm,
                        &out_rows,
                        (vocabulary, hidden),
                        "head",
                        &label,
                    )?;
                    packed.as_ref()
                }
                logits => logits,
            };
            match (logits, binding.projection, &block.shortlist, binding.shortlist) {
                (
                    ProjectionKernels::PreparedPacked(kernel),
                    ReadoutHead::Packed {
                        weight: element,
                        weight_scale,
                    }
                    | ReadoutHead::Qualified {
                        weight: element,
                        weight_scale,
                    },
                    Some(shortlist),
                    Some(coarse_element),
                ) => {
                    let k = crate::shortlist::SHORTLIST_K;
                    let coarse_weight = semantic_pattern(
                        charges,
                        coarse_element,
                        &[vocabulary, hidden],
                        "head",
                        &label,
                    )?;
                    let empty_rows = semantic_zeros(
                        charges,
                        Element::i32(),
                        &[out_rows.extents()[0], 0, vocabulary],
                        "head",
                        &label,
                    )?;
                    let no_scale = semantic_ones(charges, Element::f32(), &[0], "head", &label)?;
                    let coarse_logits = shortlist
                        .coarse
                        .first()
                        .call(charges, readout_project_rows::Args {
                            features: &projected_features,
                            weight: &coarse_weight,
                            token_rows: &empty_rows,
                            weight_scale: &no_scale,
                            softcap: 0.0,
                        })
                        .map_err(|error| {
                            qualification_dynamic("readout_project_rows(shortlist coarse)", &label, error)
                        })?
                        .value;
                    require_finite_nonzero_f32(&coarse_logits, "readout_project_rows(shortlist coarse)", &label)?;
                    let mut candidates = semantic_zeros(
                        charges,
                        Element::i32(),
                        &[out_rows.extents()[0] * k, 2],
                        "head",
                        &label,
                    )?;
                    let mut unary = semantic_zeros(
                        charges,
                        Element::f32(),
                        &[out_rows.extents()[0], k],
                        "head",
                        &label,
                    )?;
                    shortlist
                        .selection
                        .call(charges, magnitude_kernels::draft_top_k::Args {
                            logits: &coarse_logits,
                            candidates: &mut candidates,
                            unary: &mut unary,
                        })
                        .map_err(|error| qualification_dynamic("draft_top_k", &label, error))?;
                    let candidates_3d = candidates
                        .reshape(&[out_rows.extents()[0], k, 2])
                        .map_err(|error| qualification_dynamic("shortlist_token_rows", &label, error))?;
                    let token_rows = shortlist
                        .rows
                        .call(charges, magnitude_kernels::shortlist_token_rows::Args {
                            selected: &candidates_3d,
                        })
                        .map_err(|error| qualification_dynamic("shortlist_token_rows", &label, error))?
                        .value;
                    let token_rows_3d = token_rows
                        .reshape(&[out_rows.extents()[0], 1, k])
                        .map_err(|error| qualification_dynamic("readout_project_rows", &label, error))?;
                    let projection =
                        semantic_pattern(charges, element, &[vocabulary, hidden], "head", &label)?;
                    let scale = semantic_ones(
                        charges,
                        Element::f32(),
                        &[u64::from(weight_scale)],
                        "head",
                        &label,
                    )?;
                    let compact = kernel
                        .first()
                        .call(charges, readout_project_rows::Args {
                            features: &projected_features,
                            weight: &projection,
                            token_rows: &token_rows_3d,
                            weight_scale: &scale,
                            softcap: 0.0,
                        })
                        .map_err(|error| {
                            qualification_dynamic("readout_project_rows(shortlist refine)", &label, error)
                        })?
                        .value;
                    require_finite_nonzero_f32(&compact, "readout_project_rows(shortlist refine)", &label)?;
                    let mut placed = semantic_zeros(
                        charges,
                        Element::f32(),
                        &[out_rows.extents()[0], vocabulary],
                        "head",
                        &label,
                    )?;
                    shortlist
                        .placement
                        .call(charges, magnitude_kernels::place_logits_rows::Args {
                            scores: &compact,
                            token_rows: &token_rows,
                            logits: &mut placed,
                        })
                        .map_err(|error| qualification_dynamic("place_logits_rows", &label, error))?;
                }
                (
                    ProjectionKernels::PreparedPacked(kernel),
                    ReadoutHead::Packed {
                        weight: element,
                        weight_scale,
                    }
                    | ReadoutHead::Qualified {
                        weight: element,
                        weight_scale,
                    },
                    None,
                    None,
                ) => {
                    let projection =
                        semantic_pattern(charges, element, &[vocabulary, hidden], "head", &label)?;
                    let token_rows = semantic_zeros(
                        charges,
                        Element::i32(),
                        &[out_rows.extents()[0], 0, vocabulary],
                        "head",
                        &label,
                    )?;
                    let scale = semantic_ones(
                        charges,
                        Element::f32(),
                        &[u64::from(weight_scale)],
                        "head",
                        &label,
                    )?;
                    let logits = kernel
                        .first()
                        .call(charges, readout_project_rows::Args {
                            features: &projected_features,
                            weight: &projection,
                            token_rows: &token_rows,
                            weight_scale: &scale,
                            softcap: 0.0,
                        })
                        .map_err(|error| {
                            qualification_dynamic("readout_project_rows", &label, error)
                        })?
                        .value;
                    require_finite_nonzero_f32(&logits, "readout_project_rows", &label)?;
                }
                _ => {
                    return Err(qualification_dynamic(
                        "head",
                        &label,
                        "the prepared projection differs from the planned projection",
                    ));
                }
            }
        }
        Ok(())
    }
}
