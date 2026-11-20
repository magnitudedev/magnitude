use super::super::*;
use super::*;
use crate::{ReadoutHead, RecurrentForm};
use crate::operators::routed::fused_graph::{TILE_ROWS, grouped_blocks};
use crate::planning::RecordTensors;
use magnitude_family_contracts::ProgressivePlane;
use magnitude_kernels::constants::VERIFY_ROWS;

impl<'a> QualificationView<'a> {
    /// The rows `block`'s mixer reads and joins.
    fn mixer_rows(
        &self,
        block: &magnitude_family_contracts::Block,
        label: &str,
    ) -> Result<SublayerRows, CatalogFailure> {
        let paired = crate::operators::paired_block(block)
            .map_err(|error| qualification_dynamic("target", label, error))?;
        SublayerRows::of(self.geometry, Some(paired.mixer_input), label)
    }

    /// The rows `block`'s feed-forward reads and joins.
    fn feed_forward_rows(
        &self,
        block: &magnitude_family_contracts::Block,
        label: &str,
    ) -> Result<SublayerRows, CatalogFailure> {
        let paired = crate::operators::paired_block(block)
            .map_err(|error| qualification_dynamic("target", label, error))?;
        let source = paired
            .feed_forward
            .and_then(|feed_forward| feed_forward.input);
        SublayerRows::of(self.geometry, source, label)
    }

    /// Qualify the target's entries; each sited streamed block's site
    /// admission, by block index.
    pub(super) fn qualify_target(&self, charges: &seismic::Charges) -> Result<Vec<(usize, crate::assessment::Priced<()>)>, CatalogFailure> {
        let fixture = TargetFixture::new(charges, self.geometry.hidden)?;

        self.qualify_embedding(charges)?;

        for ((slot, attested), block) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
            .zip(&self.geometry.blocks)
        {
            let (MixerProgramSlot::Attention(binding), AttestedMixer::Attention(kernel)) =
                (slot.mixer(), &attested.mixer)
            else {
                continue;
            };
            let label = format!("{binding:?}");
            let rows = self.mixer_rows(block, &label)?;
            qualify_attention(charges, kernel, binding, rows, &label)?;
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
            let (MixerProgramSlot::Recurrent(binding), AttestedMixer::Recurrent(kernels)) =
                (slot.mixer(), &attested.mixer)
            else {
                continue;
            };
            let label = format!("{binding:?}");
            let rows = self.mixer_rows(block, &label)?;
            if qualified.insert((binding, rows)) {
                qualify_recurrent(charges, kernels, binding, rows, fixture.hidden, &label)?;
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
            let (MixerProgramSlot::StateSpace(binding), AttestedMixer::StateSpace(kernels)) =
                (slot.mixer(), &attested.mixer)
            else {
                continue;
            };
            if qualified.insert(binding) {
                qualify_state_space(charges, kernels, binding, &fixture.hidden_residual, &fixture.hidden_values)?;
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
                qualify_short_conv(charges, kernels, binding, &fixture.hidden_residual, &fixture.hidden_values)?;
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
            if qualified.insert((binding, *intermediate)) {
                qualify_dense(charges, kernels, binding, *intermediate, &fixture)?;
            }
        }

        self.qualify_per_layer(charges, &fixture.hidden_residual, &fixture.hidden_values)?;

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
                Some(FeedForwardProgramSlot::Routed(binding)),
                Some(AttestedFeedForward::Routed(kernels)),
            ) = (slot.feed_forward(), &attested.feed_forward)
            else {
                continue;
            };
            let rows = self.feed_forward_rows(block, &format!("{binding:?}"))?;
            if qualified.insert((binding, rows)) {
                qualify_routed(charges, binding, kernels, rows, &ExpertReads::routed(charges, binding)?)?;
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
                qualify_general_routed(charges, binding, kernels, &ExpertReads::general(charges, binding)?)?;
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
                qualify_parallel(charges, binding, kernels, &fixture.hidden_residual, &fixture.hidden_values)?;
            }
        }

        let admitted = self.qualify_banks(charges)?;
        self.qualify_exit(charges, &fixture)?;
        Ok(admitted)
    }

    /// The embedding's rows of a one-row table.
    fn qualify_embedding(&self, charges: &seismic::Charges) -> Result<(), CatalogFailure> {
        let binding = self.plan.target().embedding();
        let label = format!("{binding:?}");
        let table = semantic_pattern(
            charges,
            binding.table,
            &[1, self.geometry.hidden],
            "embedding_rows",
            &label,
        )?;
        let tokens = semantic_i32(charges, &[1, 2], &[0, 0], "embedding_rows", &label)?;
        let direct = semantic_i32(charges, &[0, 1], &[], "embedding_rows", &label)?;
        let result = self
            .programs
            .target
            .embedding
            .call(charges, embedding_rows::Args {
                table: &table,
                tokens: &tokens,
                directory: &direct,
                scale: 1.0,
                normalize: 0,
                epsilon: 0.0,
            })
            .map_err(|error| qualification_dynamic("embedding_rows", &label, error))?;
        require_finite_nonzero_f32(&result.r1, "embedding_rows", &label)
    }

    /// Each streamed layer's bank entries, and its routed form's shared
    /// entries as the bank serves them; once per binding, each sited
    /// block's stored site verdict, by block index.
    fn qualify_banks(&self, charges: &seismic::Charges) -> Result<Vec<(usize, crate::assessment::Priced<()>)>, CatalogFailure> {
        let banks = std::time::Instant::now();
        let mut qualified = std::collections::HashMap::<_, Option<crate::assessment::Priced<()>>>::new();
        let mut admitted = Vec::new();
        for (index, ((slot, attested), block)) in self
            .plan
            .target()
            .blocks()
            .iter()
            .zip(&self.programs.target.blocks)
            .zip(&self.geometry.blocks)
            .enumerate()
        {
            let (Some((banked, binding)), Some(bank), Some(feed_forward)) =
                (slot.bank(), &attested.bank, &attested.feed_forward)
            else {
                continue;
            };
            let rows = self.feed_forward_rows(block, &format!("{banked:?}"))?;
            let admission = match qualified.entry((banked, binding, rows)) {
                std::collections::hash_map::Entry::Occupied(admission) => admission.get().clone(),
                std::collections::hash_map::Entry::Vacant(vacant) => {
                    let requirement = self.profile.site_requirement(bank.site.as_ref().and_then(|site| site.form.activation.dtype()));
                    let admission = super::bank::qualify_bank(charges, banked, binding, bank, feed_forward, rows, requirement.as_ref(), self.cache)?;
                    vacant.insert(admission).clone()
                }
            };
            if let Some(admission) = admission {
                admitted.push((index, admission));
            }
        }
        tracing::info!(
            "qualified {} streamed bank bindings in {:.2} s",
            qualified.len(),
            banks.elapsed().as_secs_f64()
        );
        Ok(admitted)
    }

    /// The readout and the features, which read the decoder's exit: the
    /// final row under the exit norm, or the exit site's partials,
    /// unnormalized.
    fn qualify_exit(&self, charges: &seismic::Charges, fixture: &TargetFixture) -> Result<(), CatalogFailure> {
        let hidden = fixture.hidden;
        let exit = crate::operators::InputForm::exit(self.geometry);
        let exit_rows = semantic_ones(
            charges,
            Element::f32(),
            &[1, exit.parts * hidden],
            "target_readout",
            "exit rows",
        )?;
        self.qualify_readout(charges, exit, &exit_rows, fixture)?;

        if let (Some(binding), Some(kernel)) = (
            self.plan.target().features(),
            self.programs.target.features.as_ref(),
        ) {
            let label = format!("{binding:?}");
            let norm = semantic_ones(
                charges,
                binding.norm,
                &[exit.norm_extent(hidden)],
                "target_features",
                &label,
            )?;
            let parts = readout_parts(
                &exit_rows,
                kernel.specialization().statics(),
                "readout_features_rows",
                &label,
            )?;
            let result = kernel
                .call(charges, readout_features_rows::Args {
                    hidden: &parts,
                    norm: &norm,
                    out_rows: &fixture.out_rows,
                    epsilon: 1.0e-5,
                })
                .map_err(|error| qualification_dynamic("readout_features_rows", &label, error))?
                .value;
            require_finite_nonzero(&result, "readout_features_rows", &label)?;
        }
        Ok(())
    }

    /// The readout's features and head over `exit_rows`, the exit's rows.
    fn qualify_readout(
        &self,
        charges: &seismic::Charges,
        exit: crate::operators::InputForm,
        exit_rows: &Tensor,
        fixture: &TargetFixture,
    ) -> Result<(), CatalogFailure> {
        let (hidden, out_rows) = (fixture.hidden, &fixture.out_rows);
        let binding = self.plan.target().readout();
        let label = format!("{binding:?}");
        let norm = semantic_ones(
            charges,
            binding.norm,
            &[exit.norm_extent(hidden)],
            "target_readout",
            &label,
        )?;
        let kernel = &self.programs.target.readout.features;
        let parts = readout_parts(
            exit_rows,
            kernel.specialization().statics(),
            "readout_features_rows",
            &label,
        )?;
        let features = kernel
            .call(charges, readout_features_rows::Args {
                hidden: &parts,
                norm: &norm,
                out_rows,
                epsilon: 1.0e-5,
            })
            .map_err(|error| qualification_dynamic("readout_features_rows", &label, error))?
            .value;
        require_finite_nonzero(&features, "readout_features_rows", &label)?;
        let vocabulary = self.geometry.vocabulary;
        let readout = match &self.programs.target.readout.head {
            ReadoutHeadKernels::Qualified {
                packed,
                progressive,
                ..
            } => {
                qualify_progressive(
                    charges,
                    progressive,
                    exit_rows,
                    &norm,
                    out_rows,
                    (vocabulary, hidden),
                    "target_readout",
                    &label,
                )?;
                packed.as_ref()
            }
            readout => readout,
        };
        match (readout, binding.head) {
            (
                ReadoutHeadKernels::Packed { head, selected },
                ReadoutHead::Packed {
                    weight: element,
                    weight_scale,
                }
                | ReadoutHead::Qualified {
                    weight: element,
                    weight_scale,
                },
            ) => {
                let weight = semantic_zeros(
                    charges,
                    element,
                    &[vocabulary, hidden],
                    "target_readout",
                    &label,
                )?;
                let logits = head
                    .first()
                    .call(charges, readout_head_rows::Args {
                        hidden: exit_rows,
                        norm: &norm,
                        weight: &weight,
                        out_rows,
                        epsilon: 1.0e-5,
                        softcap: 0.0,
                        weight_scale: &semantic_ones(
                            charges,
                            Element::f32(),
                            &[weight_scale],
                            "target_readout",
                            &label,
                        )?,
                    })
                    .map_err(|error| qualification_dynamic("readout_head_rows", &label, error))?
                    .value;
                require_zero_result(&logits, "readout_head_rows", &label)?;
                let rows = semantic_i32(charges, &[1], &[0], "readout_selected_rows", &label)?;
                let selected_result = selected
                    .first()
                    .call(charges, readout_selected_rows::Args {
                        hidden: exit_rows,
                        norm: &norm,
                        weight: &weight,
                        out_rows,
                        selected: &rows,
                        epsilon: 1.0e-5,
                        softcap: 0.0,
                    })
                    .map_err(|error| {
                        qualification_dynamic("readout_selected_rows", &label, error)
                    })?
                    .value;
                require_zero_result(&selected_result, "readout_selected_rows", &label)
            }
            _ => Err(qualification_dynamic(
                "target_readout",
                &label,
                "the prepared head differs from the planned head",
            )),
        }
    }
}

/// What the target's sublayers over one plain hidden row read: the hidden
/// width, an output row list of row 0, and a residual row of ones.
struct TargetFixture {
    hidden: u64,
    out_rows: Tensor,
    hidden_values: Vec<f32>,
    hidden_residual: Tensor,
}

impl TargetFixture {
    fn new(charges: &seismic::Charges, hidden: u64) -> Result<Self, CatalogFailure> {
        let out_rows = semantic_i32(charges, &[1], &[0], "target", "out_rows")?;
        let hidden_values = vec![1.0_f32; hidden as usize];
        let hidden_residual = semantic_f32(
            charges,
            &[1, hidden],
            &hidden_values,
            "target",
            "hidden residual",
        )?;
        Ok(Self {
            hidden,
            out_rows,
            hidden_values,
            hidden_residual,
        })
    }
}

/// A dense feed-forward of `intermediate` features over the hidden row:
/// zero weights leave the residual unchanged.
fn qualify_dense(
    charges: &seismic::Charges,
    kernels: &DenseKernels,
    binding: crate::DenseBinding,
    intermediate: u64,
    fixture: &TargetFixture,
) -> Result<(), CatalogFailure> {
    let label = format!("{binding:?}");
    let (hidden, f) = (fixture.hidden, intermediate);
    let norm = semantic_zeros(charges, binding.norm, &[hidden], "target_dense", &label)?;
    let gate = semantic_zeros(charges, binding.gate, &[f, hidden], "target_dense", &label)?;
    let up = semantic_zeros(charges, binding.up, &[f, hidden], "target_dense", &label)?;
    let down = semantic_zeros(charges, binding.down, &[hidden, f], "target_dense", &label)?;
    let scale =
        |extent| semantic_ones(charges, Element::f32(), &[extent], "target_dense", &label);
    let product = kernels
        .expand
        .first()
        .call(charges, dense_expand::Args {
            residual: &fixture.hidden_residual,
            norm: &norm,
            gate_weight: &gate,
            up_weight: &up,
            out_rows: &fixture.out_rows,
            eps: 1.0e-5,
            activation: 0,
            gate_scale: &scale(binding.scales.gate)?,
            up_scale: &scale(binding.scales.up)?,
            gate_bound: &any_row_bound(charges, kernels.expand.first(), "GB", "dense_expand", &label)?,
            up_bound: &any_row_bound(charges, kernels.expand.first(), "UB", "dense_expand", &label)?,
            glu_limit: &glu_limit(charges, binding.limit, "dense_expand", &label)?,
        })
        .map_err(|error| qualification_dynamic("dense_expand", &label, error))?
        .value;
    let result = match (&kernels.output, binding.tail) {
        (SublayerOutput::Residual(output), SublayerTail::Residual) => {
            let output = output.first();
            output
                .call(charges, dense_output::Args {
                    residual: &fixture.hidden_residual,
                    product: &product,
                    down_weight: &down,
                    out_rows: &fixture.out_rows,
                    down_scale: &scale(binding.scales.down)?,
                    down_bound: &any_row_bound(charges, output, "DB", "dense_output", &label)?,
                })
                .map_err(|error| qualification_dynamic("dense_output", &label, error))?
                .value
        }
        (SublayerOutput::PostNorm(tail), SublayerTail::PostNorm { norm, .. }) => {
            qualify_post_norm(
                charges,
                tail,
                &fixture.hidden_residual,
                &product,
                &down,
                norm,
                &label,
            )?
        }
        _ => {
            return Err(qualification_dynamic(
                "dense_output",
                &label,
                "dense output entries disagree with the binding's tail",
            ));
        }
    };
    require_f32_values(&result, &fixture.hidden_values, "dense_output", &label)
}

/// What a recurrent layer's entries read: the model's recurrent geometry
/// (the entries fix it statically) over a two-bank arena, bank 0 the zero
/// seed the advance reads and bank 1 its successor, and zero weights, which
/// leave the residual unchanged.
struct RecurrentFixture {
    hidden: u64,
    norm: Tensor,
    input: Tensor,
    residual_values: Vec<f32>,
    residual: Tensor,
    qkv: Tensor,
    beta: Tensor,
    convolution: Tensor,
    rate: Tensor,
    recurrent_norm: Tensor,
    output: Tensor,
    segments: Tensor,
    stop: Tensor,
    previous_bank: Tensor,
    previous_tape: Tensor,
    following_bank: Tensor,
    window: Tensor,
    delta: Tensor,
    tape: Tensor,
    /// The banks' store, which `window`, `delta` and `tape` lie in, held
    /// while they are.
    _banks: seismic::SlabTensor,
}

impl RecurrentFixture {
    fn new(
        charges: &seismic::Charges,
        binding: RecurrentBinding,
        rows: SublayerRows,
        hidden: u64,
        label: &str,
    ) -> Result<Self, CatalogFailure> {
        let w = binding.width;
        let (key_heads, value_heads) = (binding.key_heads, binding.value_heads);
        let taps = binding.convolution_width;
        let channels = (2 * key_heads + value_heads) * w;
        let entry = "target_recurrent";
        let norm = rows.norm(charges, binding.norm, hidden, entry, label)?;
        let input = rows.input(charges, 1, hidden, entry, label)?;
        let residual_values = vec![1.0_f32; rows.residual_width(hidden) as usize];
        let residual = semantic_f32(
            charges,
            &[1, rows.residual_width(hidden)],
            &residual_values,
            entry,
            label,
        )?;
        let zeros = |element, extents: &[u64]| semantic_zeros(charges, element, extents, entry, label);
        let qkv = zeros(binding.qkv, &[channels, hidden])?;
        let beta = zeros(binding.beta, &[value_heads, hidden])?;
        let convolution = zeros(Element::f32(), &[channels, taps])?;
        let rate = zeros(Element::f32(), &[value_heads])?;
        let recurrent_norm = zeros(binding.recurrent_norm, &[w])?;
        let output = zeros(binding.output, &[hidden, value_heads * w])?;
        let segments = semantic_i32(charges, &[2, 2], &[0, 1, 1, 1], entry, label)?;
        let stop = semantic_i32(charges, &[1], &[1], entry, label)?;
        let previous_bank = semantic_i32(charges, &[1], &[0], entry, label)?;
        let previous_tape = semantic_i32(charges, &[1], &[0], entry, label)?;
        let following_bank = semantic_i32(charges, &[1], &[1], entry, label)?;
        let decays = value_heads * binding.decay_width;
        // One tape row, the least a bank holds.
        let ([window, delta, tape], banks) = zeroed_slab(
            charges,
            &[
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
                    row_shape: vec![1, (value_heads + key_heads) * w + decays],
                },
            ],
            entry,
            label,
        )?;
        Ok(Self {
            hidden,
            norm,
            input,
            residual_values,
            residual,
            qkv,
            beta,
            convolution,
            rate,
            recurrent_norm,
            output,
            segments,
            stop,
            previous_bank,
            previous_tape,
            following_bank,
            window,
            delta,
            tape,
            _banks: banks,
        })
    }
}

/// A recurrent layer reading and joining `rows`: its projection, its state
/// advance by step and by chunk, and its output, which zero weights leave
/// the residual.
fn qualify_recurrent(
    charges: &seismic::Charges,
    kernels: &RecurrentKernels,
    binding: RecurrentBinding,
    rows: SublayerRows,
    hidden: u64,
    label: &str,
) -> Result<(), CatalogFailure> {
    let mut fixture = RecurrentFixture::new(charges, binding, rows, hidden, label)?;
    let gated = match &kernels.form {
        RecurrentFormKernels::Gated {
            project,
            step,
            chunk,
        } => recurrent_gated(charges, project, step, chunk, binding, &mut fixture, label)?,
        RecurrentFormKernels::Kimi(kimi) => recurrent_kimi(charges, kimi, binding, &mut fixture, label)?,
    };
    let result = kernels
        .output
        .first()
        .call(charges, attention_output::Args {
            hidden: &fixture.residual,
            inject: &rows.inject(charges, 1, "attention_output", label)?,
            gated: &gated,
            output_weight: &fixture.output,
            output_scale: &unit_scale(charges, kernels.output.first(), "OS", "attention_output", label)?,
            output_bound: &any_row_bound(charges, kernels.output.first(), "OB", "attention_output", label)?,
        })
        .map_err(|error| qualification_dynamic("attention_output", label, error))?
        .value;
    require_f32_values(&result, &fixture.residual_values, "attention_output", label)
}

/// A Gated DeltaNet layer's projection, its step (`step`'s form) and its
/// chunk over `fixture`: the step's gated rows.
fn recurrent_gated(
    charges: &seismic::Charges,
    project: &RangedKernels<gated_delta_project::Entry>,
    step: &RecurrentStepKernels,
    chunk: &NativeKernel<gated_delta_chunk::Entry>,
    binding: RecurrentBinding,
    fixture: &mut RecurrentFixture,
    label: &str,
) -> Result<Tensor, CatalogFailure> {
    let (hidden, w, value_heads) = (fixture.hidden, binding.width, binding.value_heads);
    let (norm_epsilon, gate_function) = (f32::from_bits(binding.key_norm_epsilon), binding.gate_function);
    let zeros = |element, extents: &[u64]| semantic_zeros(charges, element, extents, "target_recurrent", label);
    let gate = zeros(binding.gate, &[value_heads * w, hidden])?;
    let alpha = zeros(binding.alpha, &[value_heads, hidden])?;
    let time_bias = zeros(Element::f32(), &[value_heads])?;
    let project = project.first();
    let projection = project
        .call(charges, gated_delta_project::Args {
            hidden: &fixture.input,
            input_norm: &fixture.norm,
            qkv_weight: &fixture.qkv,
            gate_weight: &gate,
            alpha_weight: &alpha,
            beta_weight: &fixture.beta,
            epsilon: 1.0e-5,
            qkv_bound: &any_row_bound(charges, project, "QB", "gated_delta_project", label)?,
            gate_bound: &any_row_bound(charges, project, "ZB", "gated_delta_project", label)?,
        })
        .map_err(|error| qualification_dynamic("gated_delta_project", label, error))?
        .value;
    let gated = match step {
        RecurrentStepKernels::Step(step) => {
            step.call(charges, gated_delta_step::Args {
                projection: &projection,
                convolution: &fixture.convolution,
                rate: &fixture.rate,
                time_bias: &time_bias,
                recurrent_norm: &fixture.recurrent_norm,
                segments: &fixture.segments,
                stop: &fixture.stop,
                previous_bank: &fixture.previous_bank,
                previous_tape: &fixture.previous_tape,
                following_bank: &fixture.following_bank,
                window: &mut fixture.window,
                delta: &mut fixture.delta,
                tape: &mut fixture.tape,
                norm_epsilon,
                epsilon: 1.0e-5,
                grouped: false,
                gate_function,
                slab_banks: 2,
            })
            .map_err(|error| qualification_dynamic("gated_delta_step", label, error))?
            .value
        }
        RecurrentStepKernels::Convolved { project, step } => {
            let projected = project
                .first()
                .call(charges, gated_delta_project_convolved::Args {
                    hidden: &fixture.input,
                    input_norm: &fixture.norm,
                    qkv_weight: &fixture.qkv,
                    gate_weight: &gate,
                    alpha_weight: &alpha,
                    beta_weight: &fixture.beta,
                    convolution: &fixture.convolution,
                    segments: &fixture.segments,
                    stop: &fixture.stop,
                    previous_bank: &fixture.previous_bank,
                    previous_tape: &fixture.previous_tape,
                    following_bank: &fixture.following_bank,
                    window: &mut fixture.window,
                    epsilon: 1.0e-5,
                    slab_banks: 2,
                })
                .map_err(|error| {
                    qualification_dynamic("gated_delta_project_convolved", label, error)
                })?;
            step.call(charges, gated_delta_step_convolved::Args {
                projection: &projected.r0,
                convolved: &projected.r1,
                rate: &fixture.rate,
                time_bias: &time_bias,
                recurrent_norm: &fixture.recurrent_norm,
                segments: &fixture.segments,
                stop: &fixture.stop,
                previous_bank: &fixture.previous_bank,
                previous_tape: &fixture.previous_tape,
                following_bank: &fixture.following_bank,
                delta: &mut fixture.delta,
                tape: &mut fixture.tape,
                norm_epsilon,
                epsilon: 1.0e-5,
                grouped: false,
                gate_function,
                slab_banks: 2,
            })
            .map_err(|error| {
                qualification_dynamic("gated_delta_step_convolved", label, error)
            })?
            .value
        }
    };
    chunk
        .call(charges, gated_delta_chunk::Args {
            projection: &projection,
            convolution: &fixture.convolution,
            rate: &fixture.rate,
            time_bias: &time_bias,
            recurrent_norm: &fixture.recurrent_norm,
            segments: &fixture.segments,
            stop: &fixture.stop,
            previous_bank: &fixture.previous_bank,
            previous_tape: &fixture.previous_tape,
            following_bank: &fixture.following_bank,
            window: &mut fixture.window,
            delta: &mut fixture.delta,
            tape: &mut fixture.tape,
            norm_epsilon,
            epsilon: 1.0e-5,
            grouped: false,
            gate_function,
            slab_banks: 2,
        })
        .map_err(|error| qualification_dynamic("gated_delta_chunk", label, error))?;
    Ok(gated)
}

/// A Kimi Delta Attention layer's projection, its gate's and decay's up
/// projections, and its step and chunk over `fixture`: the step's gated
/// rows.
fn recurrent_kimi(
    charges: &seismic::Charges,
    kimi: &KimiKernels,
    binding: RecurrentBinding,
    fixture: &mut RecurrentFixture,
    label: &str,
) -> Result<Tensor, CatalogFailure> {
    let RecurrentForm::PerChannel { decay_lower, gate_up, alpha_up, gate_rank, alpha_rank } = binding.form
    else {
        unreachable!("preparation gives a per-channel layer the Kimi entries")
    };
    let (hidden, w, value_heads) = (fixture.hidden, binding.width, binding.value_heads);
    let (norm_epsilon, gate_function) = (f32::from_bits(binding.key_norm_epsilon), binding.gate_function);
    let zeros = |element, extents: &[u64]| semantic_zeros(charges, element, extents, "target_recurrent", label);
    let decay_lower = f32::from_bits(decay_lower);
    let gate = zeros(binding.gate, &[gate_rank, hidden])?;
    let alpha = zeros(binding.alpha, &[alpha_rank, hidden])?;
    let time_bias = zeros(Element::f32(), &[value_heads * w])?;
    let project = kimi.project.first();
    let scale = |name| unit_scale(charges, project, name, "attention_project", label);
    let bound = |name| any_row_bound(charges, project, name, "attention_project", label);
    let projected = project
        .call(charges, attention_project::Args {
            hidden: &fixture.input,
            input_norm: &fixture.norm,
            query_weight: &fixture.qkv,
            gate_weight: &gate,
            key_weight: &alpha,
            value_weight: &fixture.beta,
            epsilon: 1.0e-5,
            project_mode: 0,
            query_scale: &scale("QS")?,
            gate_scale: &scale("GS")?,
            key_scale: &scale("KS")?,
            value_scale: &scale("VS")?,
            query_bound: &bound("QB")?,
            gate_bound: &bound("GB")?,
            key_bound: &bound("KB")?,
            value_bound: &bound("VB")?,
        })
        .map_err(|error| qualification_dynamic("attention_project", label, error))?;
    // The gate and decay inputs: the down products through
    // zero up projections.
    let expand = |kernel: &RangedKernels<project_rows::Entry>, source: &Tensor, element, rank| {
        let kernel = kernel.first();
        kernel
            .call(charges, project_rows::Args {
                source,
                weight: &zeros(element, &[value_heads * w, rank])?,
                weight_scale: &unit_scale(charges, kernel, "WS", "project_rows", label)?,
                weight_bound: &any_row_bound(charges, kernel, "WB", "project_rows", label)?,
            })
            .map(|expanded| expanded.value)
            .map_err(|error| qualification_dynamic("project_rows", label, error))
    };
    let gate_input = expand(&kimi.gate_up, &projected.r1, gate_up, gate_rank)?;
    let decay_input = expand(&kimi.alpha_up, &projected.r2, alpha_up, alpha_rank)?;
    let gated = kimi
        .step
        .call(charges, kimi_delta_step::Args {
            projection: &projected.r0,
            beta: &projected.r3,
            gate: &gate_input,
            decay: &decay_input,
            convolution: &fixture.convolution,
            rate: &fixture.rate,
            time_bias: &time_bias,
            recurrent_norm: &fixture.recurrent_norm,
            segments: &fixture.segments,
            stop: &fixture.stop,
            previous_bank: &fixture.previous_bank,
            previous_tape: &fixture.previous_tape,
            following_bank: &fixture.following_bank,
            window: &mut fixture.window,
            delta: &mut fixture.delta,
            tape: &mut fixture.tape,
            norm_epsilon,
            epsilon: 1.0e-5,
            decay_lower,
            grouped: false,
            gate_function,
            slab_banks: 2,
        })
        .map_err(|error| qualification_dynamic("kimi_delta_step", label, error))?
        .value;
    kimi.chunk
        .call(charges, kimi_delta_chunk::Args {
            projection: &projected.r0,
            beta: &projected.r3,
            gate: &gate_input,
            decay: &decay_input,
            convolution: &fixture.convolution,
            rate: &fixture.rate,
            time_bias: &time_bias,
            recurrent_norm: &fixture.recurrent_norm,
            segments: &fixture.segments,
            stop: &fixture.stop,
            previous_bank: &fixture.previous_bank,
            previous_tape: &fixture.previous_tape,
            following_bank: &fixture.following_bank,
            window: &mut fixture.window,
            delta: &mut fixture.delta,
            tape: &mut fixture.tape,
            norm_epsilon,
            epsilon: 1.0e-5,
            decay_lower,
            grouped: false,
            gate_function,
            slab_banks: 2,
        })
        .map_err(|error| qualification_dynamic("kimi_delta_chunk", label, error))?;
    Ok(gated)
}

/// Where the shared routed entries read each choice's expert: the grouped
/// entries through the map (`[X, MAP_WORDS]`) at the step (`[QS]`), the
/// records it holds for the step (each expert's by index where it is
/// empty), the stacks they read an expert from that it holds no record for;
/// the decode entries (`decode_source`) where the route's sources name,
/// `decode_record`.
pub(super) struct ExpertReads {
    pub map: Tensor,
    pub step: Tensor,
    /// The record of `records` every decode choice's sources name
    /// (SOURCE_RECORD: `XS` SOURCE_ROWS), or none, each
    /// choice reading its expert's by index (`XS` 0).
    pub decode_record: Option<i32>,
    pub records: RecordTensors<Tensor>,
    pub stacks: RecordTensors<Tensor>,
    /// The decode entries' host tier records (`TR` of them, as a bank's
    /// entries are prepared for its tier), which no source names; none:
    /// views of no records.
    pub tier: Option<RecordTensors<Tensor>>,
}

impl ExpertReads {
    /// The resident shape: zero one-expert stacks of each routed tensor's
    /// element and matrix `[N, K]`, read by index through an empty slot map.
    fn resident(
        charges: &seismic::Charges,
        record: RecordTensors<(Element, [u64; 2])>,
        label: &str,
    ) -> Result<Self, CatalogFailure> {
        let stacks = record.try_map(|_, (element, [n, k])| {
            semantic_zeros(charges, element, &[1, n, k], "target_routed", label)
        })?;
        Ok(Self {
            map: semantic_zeros(charges, Element::i32(), &[0, magnitude_kernels::constants::MAP_WORDS], "target_routed", label)?,
            step: semantic_zeros(charges, Element::i32(), &[0], "target_routed", label)?,
            decode_record: None,
            records: stacks.clone(),
            stacks,
            tier: None,
        })
    }

    /// The decode entries' `sources` of `rows` rows of `selected` choices:
    /// none (`[0, rows, selected]`) without a decode record, else each
    /// choice's the decode record as the resolve writes it
    /// (`magnitude_kernels::source_words`; none the CPU's).
    fn sources(&self, charges: &seismic::Charges, rows: u64, selected: u64, label: &str) -> Result<Tensor, CatalogFailure> {
        use magnitude_kernels::constants::SOURCE_ROWS;
        let entry = "target_routed";
        let Some(record) = self.decode_record else {
            return semantic_zeros(charges, Element::i32(), &[0, rows, selected], entry, label);
        };
        let choices = usize::try_from(rows * selected).map_err(|error| qualification_dynamic(entry, label, error))?;
        let words = magnitude_kernels::source_words(&vec![record; choices], &vec![-1; choices]);
        semantic_i32(charges, &[SOURCE_ROWS, rows, selected], &words, entry, label)
    }

    /// The decode entries' host tier records: the bank's (`Self::tier`),
    /// else none (`TR` 0), views of no records of `records`.
    pub(super) fn tier_records(&self, label: &str) -> Result<RecordTensors<Tensor>, CatalogFailure> {
        if let Some(tier) = &self.tier {
            return Ok(tier.clone());
        }
        self.records.as_ref().try_map(|_, records| {
            records.slice_leading(0, 0).map_err(|error| qualification_dynamic("target_routed", label, error))
        })
    }

    /// A fused routed binding's resident shape.
    pub(super) fn routed(charges: &seismic::Charges, binding: RoutedBinding) -> Result<Self, CatalogFailure> {
        let (h, f) = (binding.hidden, binding.features);
        let record = RecordTensors {
            gate: Some((binding.expert_gate, [f, h])),
            up: (binding.expert_up, [f, h]),
            down: (binding.expert_down, [h, f]),
        };
        Self::resident(charges, record, &format!("{binding:?}"))
    }

    /// A general routed binding's resident shape, in the experts' width.
    pub(super) fn general(
        charges: &seismic::Charges,
        binding: crate::GeneralRoutedBinding,
    ) -> Result<Self, CatalogFailure> {
        let (f, x) = (binding.shape.intermediate, binding.shape.expert_hidden);
        let record = RecordTensors {
            gate: binding.expert_gate.map(|gate| (gate, [f, x])),
            up: (binding.expert_up, [f, x]),
            down: (binding.expert_down, [x, f]),
        };
        Self::resident(charges, record, &format!("{binding:?}"))
    }
}

/// The routed entries at the model's routed geometry (they fix it
/// statically) with zero weights, so both forms leave the residual unchanged.
/// The route runs over the full expert range; the projections read expert 0
/// through all-zero routes where `reads` places it.
pub(super) fn qualify_routed(
    charges: &seismic::Charges,
    binding: RoutedBinding,
    kernels: &RoutedKernels,
    sublayer: SublayerRows,
    reads: &ExpertReads,
) -> Result<(), CatalogFailure> {
    let label = format!("{binding:?}");
    let (h, e, k, s) = (binding.hidden, binding.experts, binding.selected, binding.shared);
    let zeros = |element: Element, extents: &[u64]| {
        semantic_zeros(charges, element, extents, "target_routed", &label)
    };
    let gate = |record: &RecordTensors<Tensor>| {
        record
            .gate
            .clone()
            .ok_or_else(|| qualification_dynamic("target_routed", &label, "a fused routed record has no gate"))
    };
    let (records_gate, stacks_gate) = (gate(&reads.records)?, gate(&reads.stacks)?);
    let tier = reads.tier_records(&label)?;
    let tier_gate = gate(&tier)?;
    let norm = sublayer.norm(charges, binding.norm, h, "target_routed", &label)?;
    let router = zeros(binding.router, &[e, h])?;
    let shared_router = zeros(Element::f32(), &[h])?;
    let shared_gate = zeros(binding.shared_gate, &[s, h])?;
    let shared_up = zeros(binding.shared_up, &[s, h])?;
    let shared_down = zeros(binding.shared_down, &[h, s])?;
    let grouped_rows = VERIFY_ROWS * 2;
    for rows in [1, grouped_rows] {
        // Each projection's kernel of the range these rows lie in.
        let at = seismic::Dim::from(rows);
        let width = sublayer.residual_width(h);
        let values = vec![1.0_f32; (rows * width) as usize];
        let residual = semantic_f32(charges, &[rows, width], &values, "target_routed", &label)?;
        let input = sublayer.input(charges, rows, h, "target_routed", &label)?;
        let inject = sublayer.inject(charges, rows, "target_routed", &label)?;
        let mut routes = zeros(Element::i32(), &[rows, k])?;
        let mut scores = zeros(Element::f32(), &[rows, k])?;
        let route = |routes: &mut Tensor, scores: &mut Tensor| {
            kernels
                .route
                .call(charges, routed_route::Args {
                    residual: &input,
                    norm: &norm,
                    router: &router,
                    shared_router: &shared_router,
                    routes,
                    scores,
                    eps: 1.0e-5,
                    normalize: 1,
                    expert_scale: &unit_scale(charges, &kernels.route, "ES", "routed_route", &label)?,
                })
                .map_err(|error| qualification_dynamic("routed_route", &label, error))
        };
        let first = zeros(Element::i32(), &[rows, k])?;
        let (map, step) = (&reads.map, &reads.step);
        let result = if rows <= VERIFY_ROWS {
            let sources = reads.sources(charges, rows, k, &label)?;
            let (expert_product, shared_product, coefficient) = match &kernels.decode {
                RoutedDecodeKernels::Expand(expand) => {
                    let expand = expand.for_rows(&at);
                    let routed = route(&mut routes, &mut scores)?;
                    let expanded = expand
                        .call(charges, routed_expand::Args {
                            normalized: &routed.r0,
                            routes: &first,
                            expert_gate: &records_gate,
                            expert_up: &reads.records.up,
                            shared_gate: &shared_gate,
                            shared_up: &shared_up,
                            gate_scale: &unit_scale(charges, expand, "GS", "routed_expand", &label)?,
                            up_scale: &unit_scale(charges, expand, "US", "routed_expand", &label)?,
                        })
                        .map_err(|error| qualification_dynamic("routed_expand", &label, error))?;
                    (expanded.r0, expanded.r1, routed.r1)
                }
                RoutedDecodeKernels::SharedRoute { route, choices } => {
                    let route = route.for_rows(&at);
                    let routed = route
                        .call(charges, routed_route_shared::Args {
                            residual: &input,
                            norm: &norm,
                            router: &router,
                            shared_router: &shared_router,
                            shared_gate: &shared_gate,
                            shared_up: &shared_up,
                            routes: &mut routes,
                            scores: &mut scores,
                            eps: 1.0e-5,
                            normalize: 1,
                            expert_scale: &unit_scale(charges, route, "ES", "routed_route_shared", &label)?,
                        })
                        .map_err(|error| {
                            qualification_dynamic("routed_route_shared", &label, error)
                        })?;
                    let choices = choices.for_rows(&at);
                    let product = choices
                        .call(charges, routed_gate_up::Args {
                            normalized: &routed.r0,
                            routes: &first,
                            sources: &sources,
                            expert_gate: &records_gate,
                            expert_up: &reads.records.up,
                            tier_gate: &tier_gate,
                            tier_up: &tier.up,
                            activation: 0,
                            gate_scale: &unit_scale(charges, choices, "GS", "routed_gate_up", &label)?,
                            up_scale: &unit_scale(charges, choices, "US", "routed_gate_up", &label)?,
                            glu_limit: &glu_limit(charges, None, "routed_gate_up", &label)?,
                        })
                        .map_err(|error| qualification_dynamic("routed_gate_up", &label, error))?
                        .value;
                    (product, routed.r2, routed.r1)
                }
            };
            kernels
                .output
                .for_rows(&at)
                .call(charges, routed_output::Args {
                    residual: &residual,
                    inject: &inject,
                    expert_product: &expert_product,
                    shared_product: &shared_product,
                    routes: &first,
                    sources: &sources,
                    scores: &scores,
                    coefficient: &coefficient,
                    expert_down: &reads.records.down,
                    tier_down: &tier.down,
                    shared_down: &shared_down,
                    operand_norm: &seismic::Tensor::from_host(charges, binding.norm, &[0], &[])
                        .map_err(|error| qualification_dynamic("routed_output", &label, error))?,
                    operand_epsilon: 0.0,
                })
                .map_err(|error| qualification_dynamic("routed_output", &label, error))?
                .value
        } else {
            let routed = route(&mut routes, &mut scores)?;
            let blocks = grouped_blocks(rows, e, k)
                .map_err(|error| qualification_dynamic("routed_group", &label, error))?
                .fixed();
            let grouped = semantic_i32(charges, &[1], &[rows as i32], "routed_group", &label)?;
            let from_row = semantic_i32(charges, &[1], &[0], "routed_group", &label)?;
            let mut counts = zeros(Element::i32(), &[e])?;
            let mut order = zeros(Element::i32(), &[blocks, TILE_ROWS])?;
            let mut inverse = zeros(Element::i32(), &[rows, k])?;
            let mut block_experts = zeros(Element::i32(), &[blocks])?;
            let mut used = zeros(Element::i32(), &[1])?;
            kernels
                .group
                .call(charges, routed_group::Args {
                    first: &from_row,
                    rows: &grouped,
                    routes: &first,
                    counts: &mut counts,
                    order: &mut order,
                    inverse: &mut inverse,
                    blocks: &mut block_experts,
                    used: &mut used,
                })
                .map_err(|error| qualification_dynamic("routed_group", &label, error))?;
            let experts = kernels
                .experts
                .for_rows(&at)
                .call(charges, routed_experts::Args {
                    first: &from_row,
                    rows: &grouped,
                    normalized: &routed.r0,
                    order: &order,
                    blocks: &block_experts,
                    used: &used,
                    expert_gate: &stacks_gate,
                    expert_up: &reads.stacks.up,
                    expert_down: &reads.stacks.down,
                    map,
                    step,
                    records_gate: &records_gate,
                    records_up: &reads.records.up,
                    records_down: &reads.records.down,
                    activation: 0,
                    gate_scale: &unit_scale(charges, kernels.experts.for_rows(&at), "GS", "routed_experts", &label)?,
                    up_scale: &unit_scale(charges, kernels.experts.for_rows(&at), "US", "routed_experts", &label)?,
                    glu_limit: &glu_limit(charges, None, "routed_experts", &label)?,
                })
                .map_err(|error| qualification_dynamic("routed_experts", &label, error))?
                .value;
            kernels
                .combine
                .for_rows(&at)
                .call(charges, routed_combine::Args {
                    residual: &residual,
                    inject: &inject,
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

/// A parallel sublayer: its routed branch as a general routed sublayer, then
/// the dense branch over zero weights and `moe_tail` over zero branch
/// outputs, which leaves the residual unchanged.
fn qualify_parallel(
    charges: &seismic::Charges,
    binding: crate::ParallelBinding,
    kernels: &ParallelKernels,
    hidden_residual: &Tensor,
    hidden_values: &[f32],
) -> Result<(), CatalogFailure> {
    qualify_general_routed(
        charges,
        binding.routed,
        &kernels.routed,
        &ExpertReads::general(charges, binding.routed)?,
    )?;
    let label = format!("{binding:?}");
    let hidden = binding.hidden;
    let dense = binding.dense;
    let zeros = |element: Element, extents: &[u64], entry: &'static str| {
        semantic_zeros(charges, element, extents, entry, &label)
    };
    let features = dense.features;
    let out_rows = semantic_i32(charges, &[1], &[0], "dense_expand", &label)?;
    let product = kernels
        .expand
        .first()
        .call(charges, dense_expand::Args {
            residual: hidden_residual,
            norm: &zeros(dense.norm, &[hidden], "dense_expand")?,
            gate_weight: &zeros(dense.gate, &[features, hidden], "dense_expand")?,
            up_weight: &zeros(dense.up, &[features, hidden], "dense_expand")?,
            out_rows: &out_rows,
            eps: 1.0e-5,
            activation: 0,
            gate_scale: &unit_scale(charges, kernels.expand.first(), "GS", "dense_expand", &label)?,
            up_scale: &unit_scale(charges, kernels.expand.first(), "US", "dense_expand", &label)?,
            gate_bound: &any_row_bound(charges, kernels.expand.first(), "GB", "dense_expand", &label)?,
            up_bound: &any_row_bound(charges, kernels.expand.first(), "UB", "dense_expand", &label)?,
            glu_limit: &glu_limit(charges, dense.limit, "dense_expand", &label)?,
        })
        .map_err(|error| qualification_dynamic("dense_expand", &label, error))?
        .value;
    let projected = kernels
        .down
        .first()
        .call(charges, project_rows::Args {
            source: &product,
            weight: &zeros(dense.down, &[hidden, features], "project_rows")?,
            weight_scale: &unit_scale(charges, kernels.down.first(), "WS", "project_rows", &label)?,
            weight_bound: &any_row_bound(charges, kernels.down.first(), "WB", "project_rows", &label)?,
        })
        .map_err(|error| qualification_dynamic("project_rows", &label, error))?
        .value;
    let branch = zeros(Element::f32(), &[1, hidden], "moe_tail")?;
    let norm = zeros(binding.norm, &[hidden], "moe_tail")?;
    let result = kernels
        .tail
        .call(charges, moe_tail::Args {
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

/// A general routed sublayer at the model's geometry with zero weights, at
/// one decode row and one grouped row class: every choice routes to expert
/// 0, read where `reads` places it, the shared expert and latent
/// projections are zero, so the residual comes back unchanged.
pub(super) fn qualify_general_routed(
    charges: &seismic::Charges,
    binding: crate::GeneralRoutedBinding,
    kernels: &GeneralRoutedKernels,
    reads: &ExpertReads,
) -> Result<(), CatalogFailure> {
    let label = format!("{binding:?}");
    let entry = "target_general_routed";
    // Each dense entry's scale ports at their bound extents.
    let scale = |extent| semantic_ones(charges, Element::f32(), &[extent], entry, &label);
    let scales = binding.scales;
    let shape = binding.shape;
    let (h, e, k, x) = (shape.hidden, shape.experts, shape.selected, shape.expert_hidden);
    let zeros =
        |element: Element, extents: &[u64]| semantic_zeros(charges, element, extents, entry, &label);
    macro_rules! failed {
        ($name:literal) => {
            |error| qualification_dynamic($name, &label, error)
        };
    }
    let norm = zeros(binding.norm, &[h])?;
    let router_norm = zeros(binding.router_norm, &[h])?;
    let router = zeros(binding.router, &[e, h])?;
    let bias = zeros(Element::f32(), &[e])?;
    // A gated entry's gate: the record's, or of an up-only record its up's.
    let gate = |record: &RecordTensors<Tensor>| record.gate.clone().unwrap_or_else(|| record.up.clone());
    let (records_gate, stacks_gate) = (gate(&reads.records), gate(&reads.stacks));
    let tier = reads.tier_records(&label)?;
    let tier_gate = gate(&tier);
    let (map, step) = (&reads.map, &reads.step);
    for rows in [1, VERIFY_ROWS * 2] {
        // Each projection's kernel of the range these rows lie in.
        let at = seismic::Dim::from(rows);
        let values = vec![1.0_f32; (rows * h) as usize];
        let residual = semantic_f32(charges, &[rows, h], &values, entry, &label)?;
        let out_rows = semantic_i32(
            charges,
            &[rows],
            &(0..rows as i32).collect::<Vec<_>>(),
            entry,
            &label,
        )?;
        let base = match (&kernels.shared, shape.shared, binding.shared) {
            (Some((expansion, output)), Some((s, expansion_shape)), Some((gate, up, down))) => {
                let up = zeros(up, &[s, h])?;
                let product = match (expansion, gate) {
                    (DenseExpansionKernel::Gated(kernel), Some(gate)) => {
                        let kernel = kernel.for_rows(&at);
                        kernel
                            .call(charges, dense_expand::Args {
                                residual: &residual,
                                norm: &norm,
                                gate_weight: &zeros(gate, &[s, h])?,
                                up_weight: &up,
                                out_rows: &out_rows,
                                eps: 1.0e-5,
                                activation: expansion_shape.activation,
                                gate_scale: &scale(scales.shared.gate)?,
                                up_scale: &scale(scales.shared.up)?,
                                gate_bound: &any_row_bound(charges, kernel, "GB", "dense_expand", &label)?,
                                up_bound: &any_row_bound(charges, kernel, "UB", "dense_expand", &label)?,
                                glu_limit: &glu_limit(charges, expansion_shape.limit, "dense_expand", &label)?,
                            })
                            .map_err(failed!("dense_expand"))?
                            .value
                    }
                    (DenseExpansionKernel::Plain(kernel), None) => {
                        kernel
                            .for_rows(&at)
                            .call(charges, dense_up::Args {
                                residual: &residual,
                                norm: &norm,
                                up_weight: &up,
                                out_rows: &out_rows,
                                eps: 1.0e-5,
                                activation: expansion_shape.activation,
                                up_scale: &scale(scales.shared.up)?,
                            })
                            .map_err(failed!("dense_up"))?
                            .value
                    }
                    _ => {
                        return Err(qualification_dynamic(
                            "dense_output",
                            &label,
                            "shared expert entries disagree with the binding",
                        ));
                    }
                };
                let output = output.for_rows(&at);
                output
                    .call(charges, dense_output::Args {
                        residual: &residual,
                        product: &product,
                        down_weight: &zeros(down, &[h, s])?,
                        out_rows: &out_rows,
                        down_scale: &scale(scales.shared.down)?,
                        down_bound: &any_row_bound(charges, output, "DB", "dense_output", &label)?,
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
                ));
            }
        };
        let mut routes = zeros(Element::i32(), &[rows, k])?;
        let mut weights = zeros(Element::f32(), &[rows, k])?;
        let normalized = kernels
            .select
            .call(charges, routed_select::Args {
                residual: &residual,
                norm: &norm,
                router_norm: &router_norm,
                router: &router,
                bias: &bias,
                expert_scale: &unit_scale(charges, &kernels.select, "ES", "routed_select", &label)?,
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
                    .for_rows(&at)
                    .call(charges, project_rows::Args {
                        source: &normalized,
                        weight: &zeros(down, &[x, h])?,
                        weight_scale: &scale(scales.latent.0)?,
                        weight_bound: &any_row_bound(charges, project.for_rows(&at), "WB", "project_rows", &label)?,
                    })
                    .map_err(failed!("project_rows"))?
                    .value,
                zeros(Element::f32(), &[rows, x])?,
            ),
            _ => (normalized, base.clone()),
        };
        // Every choice routes to expert 0.
        let first = zeros(Element::i32(), &[rows, k])?;
        let routed = if rows <= VERIFY_ROWS {
            let sources = reads.sources(charges, rows, k, &label)?;
            let product = match &kernels.experts {
                ExpertKernels::Gated { decode, .. } => {
                    let decode = decode.for_rows(&at);
                    decode
                        .call(charges, routed_gate_up::Args {
                            normalized: &input,
                            routes: &first,
                            sources: &sources,
                            expert_gate: &records_gate,
                            expert_up: &reads.records.up,
                            tier_gate: &tier_gate,
                            tier_up: &tier.up,
                            activation: shape.experts_expansion.activation,
                            gate_scale: &unit_scale(charges, decode, "GS", "routed_gate_up", &label)?,
                            up_scale: &unit_scale(charges, decode, "US", "routed_gate_up", &label)?,
                            glu_limit: &glu_limit(charges, shape.experts_expansion.limit, "routed_gate_up", &label)?,
                        })
                        .map_err(failed!("routed_gate_up"))?
                        .value
                }
                ExpertKernels::Plain { decode, .. } => {
                    let decode = decode.for_rows(&at);
                    decode
                        .call(charges, routed_up::Args {
                            normalized: &input,
                            routes: &first,
                            sources: &sources,
                            expert_up: &reads.records.up,
                            tier_up: &tier.up,
                            up_scale: &unit_scale(charges, decode, "US", "routed_up", &label)?,
                            activation: shape.experts_expansion.activation,
                        })
                        .map_err(failed!("routed_up"))?
                        .value
                }
            };
            kernels
                .down
                .for_rows(&at)
                .call(charges, routed_down::Args {
                    base: &sum_base,
                    product: &product,
                    routes: &first,
                    sources: &sources,
                    weights: &weights,
                    expert_down: &reads.records.down,
                    tier_down: &tier.down,
                    policy: &semantic_i32(
                        charges,
                        &[magnitude_kernels::constants::POLICY_WORDS],
                        &[0; magnitude_kernels::constants::POLICY_WORDS as usize],
                        "routed_down",
                        &label,
                    )?,
                })
                .map_err(failed!("routed_down"))?
                .value
        } else {
            let blocks = grouped_blocks(rows, e, k).map_err(failed!("routed_group"))?.fixed();
            let grouped_rows = semantic_i32(charges, &[1], &[rows as i32], "routed_group", &label)?;
            let from_row = semantic_i32(charges, &[1], &[0], "routed_group", &label)?;
            let mut counts = zeros(Element::i32(), &[e])?;
            let mut order = zeros(Element::i32(), &[blocks, TILE_ROWS])?;
            let mut inverse = zeros(Element::i32(), &[rows, k])?;
            let mut block_experts = zeros(Element::i32(), &[blocks])?;
            let mut used = zeros(Element::i32(), &[1])?;
            kernels
                .group
                .call(charges, routed_group::Args {
                    first: &from_row,
                    rows: &grouped_rows,
                    routes: &first,
                    counts: &mut counts,
                    order: &mut order,
                    inverse: &mut inverse,
                    blocks: &mut block_experts,
                    used: &mut used,
                })
                .map_err(failed!("routed_group"))?;
            let expert_output = match &kernels.experts {
                ExpertKernels::Gated { grouped, .. } => {
                    let grouped = grouped.for_rows(&at);
                    grouped
                        .call(charges, routed_experts::Args {
                            first: &from_row,
                            rows: &grouped_rows,
                            normalized: &input,
                            order: &order,
                            blocks: &block_experts,
                            used: &used,
                            expert_gate: &stacks_gate,
                            expert_up: &reads.stacks.up,
                            expert_down: &reads.stacks.down,
                            map,
                            step,
                            records_gate: &records_gate,
                            records_up: &reads.records.up,
                            records_down: &reads.records.down,
                            activation: shape.experts_expansion.activation,
                            gate_scale: &unit_scale(charges, grouped, "GS", "routed_experts", &label)?,
                            up_scale: &unit_scale(charges, grouped, "US", "routed_experts", &label)?,
                            glu_limit: &glu_limit(charges, shape.experts_expansion.limit, "routed_experts", &label)?,
                        })
                        .map_err(failed!("routed_experts"))?
                        .value
                }
                ExpertKernels::Plain { grouped, .. } => {
                    let grouped = grouped.for_rows(&at);
                    grouped
                        .call(charges, routed_experts_up::Args {
                            first: &from_row,
                            rows: &grouped_rows,
                            normalized: &input,
                            order: &order,
                            blocks: &block_experts,
                            used: &used,
                            expert_up: &reads.stacks.up,
                            expert_down: &reads.stacks.down,
                            map,
                            step,
                            records_up: &reads.records.up,
                            records_down: &reads.records.down,
                            up_scale: &unit_scale(charges, grouped, "US", "routed_experts_up", &label)?,
                            activation: shape.experts_expansion.activation,
                        })
                        .map_err(failed!("routed_experts_up"))?
                        .value
                }
            };
            kernels
                .scatter
                .call(charges, routed_scatter::Args {
                    base: &sum_base,
                    expert_output: &expert_output,
                    inverse: &inverse,
                    weights: &weights,
                })
                .map_err(failed!("routed_scatter"))?
                .value
        };
        let result = match (&kernels.latent, binding.latent) {
            (Some((_, up_kernel)), Some((_, up))) => {
                let up_kernel = up_kernel.for_rows(&at);
                up_kernel
                    .call(charges, dense_output::Args {
                        residual: &base,
                        product: &routed,
                        down_weight: &zeros(up, &[h, x])?,
                        out_rows: &out_rows,
                        down_scale: &scale(scales.latent.1)?,
                        down_bound: &any_row_bound(charges, up_kernel, "DB", "dense_output", &label)?,
                    })
                    .map_err(failed!("dense_output"))?
                    .value
            }
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
    charges: &seismic::Charges,
    kernels: &ShortConvKernels,
    binding: crate::ShortConvBinding,
    hidden_residual: &Tensor,
    hidden_values: &[f32],
) -> Result<(), CatalogFailure> {
    let label = format!("{binding:?}");
    let entry = "target_short_conv";
    let shape = binding.shape;
    let zeros = |element, extents: &[u64]| semantic_zeros(charges, element, extents, entry, &label);
    let norm = zeros(binding.norm, &[shape.hidden])?;
    let input_gate = zeros(binding.input_gate, &[shape.channels, shape.hidden])?;
    let output_gate = zeros(binding.output_gate, &[shape.channels, shape.hidden])?;
    let value = zeros(binding.value, &[shape.channels, shape.hidden])?;
    let convolution = zeros(Element::f32(), &[shape.channels, shape.convolution_width])?;
    let output = zeros(binding.output, &[shape.hidden, shape.channels])?;
    let segments = semantic_i32(charges, &[2, 2], &[0, 1, 1, 1], entry, &label)?;
    let stop = semantic_i32(charges, &[1], &[1], entry, &label)?;
    let previous_bank = semantic_i32(charges, &[1], &[0], entry, &label)?;
    let previous_tape = semantic_i32(charges, &[1], &[0], entry, &label)?;
    let following_bank = semantic_i32(charges, &[1], &[1], entry, &label)?;
    // One tape row, the least a bank holds.
    let (regions, _banks) = zeroed_slab(
        charges,
        &[seismic::SlabRegion {
            element: Element::f32(),
            row_shape: vec![shape.convolution_width, shape.channels],
        }],
        entry,
        &label,
    )?;
    let [mut window] = regions;
    let projection = kernels
        .project
        .first()
        .call(charges, short_conv_project::Args {
            residual: hidden_residual,
            norm: &norm,
            b_weight: &input_gate,
            c_weight: &output_gate,
            x_weight: &value,
            eps: 1.0e-5,
            b_scale: &unit_scale(charges, kernels.project.first(), "BS", entry, &label)?,
            c_scale: &unit_scale(charges, kernels.project.first(), "CS", entry, &label)?,
            x_scale: &unit_scale(charges, kernels.project.first(), "XS", entry, &label)?,
        })
        .map_err(|error| qualification_dynamic("short_conv_project", &label, error))?
        .value;
    let gated = kernels
        .rows
        .call(charges, short_conv_rows::Args {
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
        .value;
    let result = match (&kernels.output, binding.tail) {
        (SublayerOutput::Residual(output_kernel), SublayerTail::Residual) => output_kernel
            .first()
            .call(charges, attention_output::Args {
                hidden: hidden_residual,
                inject: &semantic_zeros(charges, Element::f32(), &[1, 0], "attention_output", &label)?,
                gated: &gated
                    .reshape(&[1, 1, shape.channels])
                    .map_err(|error| qualification_dynamic("short_conv_rows", &label, error))?,
                output_weight: &output,
                output_scale: &unit_scale(charges, output_kernel.first(), "OS", "attention_output", &label)?,
                output_bound: &any_row_bound(charges, output_kernel.first(), "OB", "attention_output", &label)?,
            })
            .map_err(|error| qualification_dynamic("attention_output", &label, error))?
            .value,
        (SublayerOutput::PostNorm(tail), SublayerTail::PostNorm { norm, .. }) => {
            qualify_post_norm(charges, tail, hidden_residual, &gated, &output, norm, &label)?
        }
        _ => {
            return Err(qualification_dynamic(
                "attention_output",
                &label,
                "short convolution output entries disagree with the binding's tail",
            ));
        }
    };
    require_f32_values(&result, hidden_values, "attention_output", &label)
}

/// One row of a state-space layer at the model's geometry (the entries fix
/// it statically) over a two-bank arena: bank 0 is the zero seed the advance
/// reads, bank 1 its successor. Zero weights leave the residual unchanged.
fn qualify_state_space(
    charges: &seismic::Charges,
    kernels: &StateSpaceKernels,
    binding: crate::StateSpaceBinding,
    hidden_residual: &Tensor,
    hidden_values: &[f32],
) -> Result<(), CatalogFailure> {
    let label = format!("{binding:?}");
    let entry = "target_state_space";
    let shape = binding.shape;
    let zeros = |element, extents: &[u64]| semantic_zeros(charges, element, extents, entry, &label);
    let f32 = Element::f32();
    // A unit accumulator scale of the port's extent (absent at 0).
    let scale = |extent| semantic_ones(charges, f32, &[extent], entry, &label);
    let norm = zeros(binding.norm, &[shape.hidden])?;
    let projection_weight = zeros(
        binding.projection,
        &[shape.projection_width(), shape.hidden],
    )?;
    let empty = projection_weight
        .slice_leading(0, 0)
        .map_err(|error| qualification_dynamic(entry, &label, error))?;
    let convolution = zeros(f32, &[shape.channels(), shape.convolution_width])?;
    let convolution_bias = zeros(f32, &[shape.channels()])?;
    let rate = zeros(f32, &[shape.heads])?;
    let time_bias = zeros(f32, &[shape.heads])?;
    let skip = zeros(f32, &[shape.heads])?;
    let state_norm = zeros(
        f32,
        &[shape.norm_groups(), shape.norm_heads, shape.head_width],
    )?;
    let output = zeros(binding.output, &[shape.hidden, shape.inner()])?;
    let segments = semantic_i32(charges, &[2, 2], &[0, 1, 1, 1], entry, &label)?;
    let stop = semantic_i32(charges, &[1], &[1], entry, &label)?;
    let previous_bank = semantic_i32(charges, &[1], &[0], entry, &label)?;
    let previous_tape = semantic_i32(charges, &[1], &[0], entry, &label)?;
    let following_bank = semantic_i32(charges, &[1], &[1], entry, &label)?;
    // One tape row, the least a bank holds.
    let (regions, _banks) = zeroed_slab(
        charges,
        &[
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
        entry,
        &label,
    )?;
    let [mut window, mut state, mut tape] = regions;
    let projection = kernels
        .project
        .first()
        .call(charges, attention_project::Args {
            hidden: hidden_residual,
            input_norm: &norm,
            query_weight: &projection_weight,
            gate_weight: &empty,
            key_weight: &empty,
            value_weight: &empty,
            epsilon: 1.0e-5,
            project_mode: 0,
            query_scale: &scale(binding.scales.projection)?,
            gate_scale: &scale(0)?,
            key_scale: &scale(0)?,
            value_scale: &scale(0)?,
            query_bound: &any_row_bound(charges, kernels.project.first(), "QB", "attention_project", &label)?,
            gate_bound: &any_row_bound(charges, kernels.project.first(), "GB", "attention_project", &label)?,
            key_bound: &any_row_bound(charges, kernels.project.first(), "KB", "attention_project", &label)?,
            value_bound: &any_row_bound(charges, kernels.project.first(), "VB", "attention_project", &label)?,
        })
        .map_err(|error| qualification_dynamic("attention_project", &label, error))?
        .r0;
    macro_rules! advance {
        ($module:ident, $kernel:expr) => {
            $kernel
                .call(charges, $module::Args {
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
        .call(charges, state_space_gate::Args {
            mixed: &mixed,
            projection: &projection,
            state_norm: &state_norm,
            epsilon: 1.0e-5,
        })
        .map_err(|error| qualification_dynamic("state_space_gate", &label, error))?
        .value;
    let result = kernels
        .output
        .first()
        .call(charges, attention_output::Args {
            hidden: hidden_residual,
            inject: &semantic_zeros(charges, Element::f32(), &[1, 0], "attention_output", &label)?,
            gated: &gated,
            output_weight: &output,
            output_scale: &scale(binding.scales.output)?,
            output_bound: &any_row_bound(charges, kernels.output.first(), "OB", "attention_output", &label)?,
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
        charges: &seismic::Charges,
        hidden_residual: &Tensor,
        hidden_values: &[f32],
    ) -> Result<(), CatalogFailure> {
        let hidden = self.geometry.hidden;
        if let (Some(binding), Some(kernels), Some(entry)) = (
            self.plan.target().per_layer(),
            &self.programs.target.per_layer,
            &self.geometry.entry.per_layer,
        ) {
            let label = format!("{binding:?}");
            let channels = binding.layers * binding.width;
            let embedded = semantic_f32(
                charges,
                &[1, 1, hidden],
                hidden_values,
                "per_layer_entry",
                &label,
            )?;
            let rounded = kernels
                .round
                .call(charges, import_dense::Args { source: &embedded })
                .map_err(|error| qualification_dynamic("import_dense", &label, error))?
                .value;
            let projection = semantic_zeros(
                charges,
                binding.projection,
                &[channels, hidden],
                "per_layer_entry",
                &label,
            )?;
            let projected = kernels
                .project
                .first()
                .call(charges, project_rows::Args {
                    source: &rounded
                        .reshape(&[1, hidden])
                        .map_err(|error| qualification_dynamic("project_rows", &label, error))?,
                    weight: &projection,
                    weight_scale: &unit_scale(charges, kernels.project.first(), "WS", "project_rows", &label)?,
                    weight_bound: &any_row_bound(charges, kernels.project.first(), "WB", "project_rows", &label)?,
                })
                .map_err(|error| qualification_dynamic("project_rows", &label, error))?
                .value;
            let source = semantic_zeros(
                charges,
                binding.table_source,
                &[1, 1, channels],
                "per_layer_entry",
                &label,
            )?;
            let table = match &kernels.table {
                TableConversion::Dense(kernel) => {
                    kernel
                        .call(charges, import_dense::Args { source: &source })
                        .map_err(|error| qualification_dynamic("import_dense", &label, error))?
                        .value
                }
                TableConversion::Repack(kernel) => {
                    kernel
                        .call(charges, repack_weight::Args { source: &source })
                        .map_err(|error| qualification_dynamic("repack_weight", &label, error))?
                        .value
                }
            };
            let norm = semantic_zeros(
                charges,
                binding.norm,
                &[binding.width],
                "per_layer_inputs",
                &label,
            )?;
            let combined = kernels
                .inputs
                .call(charges, per_layer_inputs::Args {
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
            let gate = semantic_zeros(
                charges,
                binding.gate,
                &[width, hidden],
                "per_layer_gate",
                &label,
            )?;
            let inputs = semantic_zeros(
                charges,
                Element::f32(),
                &[1, layers, width],
                "per_layer_gate",
                &label,
            )?;
            let gated = kernels
                .gate
                .first()
                .call(charges, per_layer_gate::Args {
                    hidden: hidden_residual,
                    gate_weight: &gate,
                    inputs: &inputs,
                    layer: 0,
                    activation: 1,
                    gate_scale: &unit_scale(charges, kernels.gate.first(), "GS", "per_layer_gate", &label)?,
                })
                .map_err(|error| qualification_dynamic("per_layer_gate", &label, error))?
                .value;
            let projection = semantic_zeros(
                charges,
                binding.projection,
                &[hidden, width],
                "per_layer_gate",
                &label,
            )?;
            let result = qualify_post_norm(
                charges,
                &kernels.output,
                hidden_residual,
                &gated,
                &projection,
                norm,
                &label,
            )?;
            require_f32_values(&result, hidden_values, "per_layer_gate", &label)?;
        }
        Ok(())
    }
}

/// A progressive head's levels and full exact pass over zero planes of
/// `[vocabulary, hidden]`: every logit of every level is zero, every row
/// reaches the zero threshold, and the exact level keeps them.
#[allow(clippy::too_many_arguments)]
pub(super) fn qualify_progressive(
    charges: &seismic::Charges,
    kernels: &ProgressiveReadoutKernels,
    hidden_residual: &Tensor,
    norm: &Tensor,
    out_rows: &Tensor,
    (vocabulary, hidden): (u64, u64),
    context: &'static str,
    label: &str,
) -> Result<(), CatalogFailure> {
    let label = label.to_owned();
    let partials = readout_parts(
        hidden_residual,
        kernels.top.specialization().statics(),
        "readout_top_rows",
        &label,
    )?;
    let hidden_residual = &partials;
    let depth = match kernels.top.specialization().statics().get("P") {
        Some(1) => magnitude_family_contracts::ProgressiveCodeDepth::Four,
        Some(3) => magnitude_family_contracts::ProgressiveCodeDepth::Eight,
        _ => {
            return Err(CatalogFailure::Preparation {
                entry: context,
                bindings: label.clone(),
                outcome: "progressive code depth is absent".into(),
            });
        }
    };
    let plane = |plane: ProgressivePlane| {
        semantic_zeros(
            charges,
            crate::progressive::element(plane),
            &plane.shape(vocabulary, hidden, depth),
            context,
            &label,
        )
    };
    let top = plane(ProgressivePlane::Top)?;
    let bit3 = plane(ProgressivePlane::Next)?;
    let rest = plane(ProgressivePlane::Rest)?;
    let scales = plane(ProgressivePlane::Scales)?;
    let radius = plane(ProgressivePlane::Radius)?;
    let draws = semantic_zeros(charges, Element::u32(), &[1, 6], context, &label)?;
    let temperature = semantic_ones(charges, Element::f32(), &[1], context, &label)?;
    let mask = semantic_zeros(
        charges,
        Element::u32(),
        &[1, vocabulary.div_ceil(32)],
        context,
        &label,
    )?;
    let constrained = semantic_i32(charges, &[1], &[0], context, &label)?;
    let first = kernels
        .top
        .call(charges, readout_top_rows::Args {
            softcap: 0.0,
            hidden: hidden_residual,
            norm,
            top: &top,
            bit3: &bit3,
            rest: &rest,
            scales: &scales,
            radius: &radius,
            out_rows,
            draws: &draws,
            temperature: &temperature,
            mask: &mask,
            constrained: &constrained,
            epsilon: 1.0e-5,
        })
        .map_err(|error| qualification_dynamic("readout_top_rows", &label, error))?;
    require_zero_result(&first.r0, "readout_top_rows", &label)?;
    let second = kernels
        .refine
        .call(charges, readout_refine_rows::Args {
            softcap: 0.0,
            features: &first.r2,
            top: &top,
            bit3: &bit3,
            rest: &rest,
            scales: &scales,
            radius: &radius,
            coarse: &first.r0,
            floor: &first.r1,
            length: &first.r3,
            draws: &draws,
            temperature: &temperature,
            mask: &mask,
            constrained: &constrained,
        })
        .map_err(|error| qualification_dynamic("readout_refine_rows", &label, error))?;
    require_zero_result(&second.r0, "readout_refine_rows", &label)?;
    let exact = kernels
        .exact
        .call(charges, readout_exact_rows::Args {
            softcap: 0.0,
            features: &first.r2,
            top: &top,
            bit3: &bit3,
            rest: &rest,
            scales: &scales,
            radius: &radius,
            fine: &second.r0,
            floor: &second.r1,
            length: &first.r3,
            draws: &draws,
            temperature: &temperature,
            mask: &mask,
            constrained: &constrained,
        })
        .map_err(|error| qualification_dynamic("readout_exact_rows", &label, error))?
        .value;
    require_zero_result(&exact, "readout_exact_rows", &label)?;
    let planes = kernels
        .planes
        .call(charges, readout_planes_rows::Args {
            softcap: 0.0,
            hidden: hidden_residual,
            norm,
            top: &top,
            bit3: &bit3,
            rest: &rest,
            scales: &scales,
            out_rows,
            epsilon: 1.0e-5,
        })
        .map_err(|error| qualification_dynamic("readout_planes_rows", &label, error))?
        .value;
    require_zero_result(&planes, "readout_planes_rows", &label)?;
    Ok(())
}
