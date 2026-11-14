use super::super::*;
use crate::AttentionValues;
use seismic::{Layout, OperandRecipe};

pub(super) fn qualification(
    entry: &'static str,
    bindings: &'static str,
    outcome: impl fmt::Display,
) -> CatalogFailure {
    CatalogFailure::Qualification {
        entry,
        bindings: bindings.into(),
        outcome: outcome.to_string(),
    }
}

pub(super) fn qualification_dynamic(
    entry: &'static str,
    bindings: &str,
    outcome: impl fmt::Display,
) -> CatalogFailure {
    CatalogFailure::Qualification {
        entry,
        bindings: bindings.to_owned(),
        outcome: outcome.to_string(),
    }
}

/// A sublayer's rows as its block graph forms and joins them
/// (`operators::block`): its consumer entry forms its input from `form`'s
/// parts (`PT`, `NM`), and its output entry joins `streams` residual streams
/// (`SN`), scattering with one inject value per stream when `injected`
/// (`IN`). One residual stream is one normalized row, joined by a plain add.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct SublayerRows {
    pub form: crate::operators::InputForm,
    pub streams: u64,
    /// The width `IN` of a row of its join values (`operators::Join`).
    pub width: u64,
}

impl SublayerRows {
    pub(super) const SINGLE: Self = Self {
        form: crate::operators::InputForm::SINGLE,
        streams: 1,
        width: 0,
    };

    /// The rows of a target sublayer of `decoder` reading `source`: a mix
    /// site's sublayer reads its site's partials, or a Sinkhorn site's
    /// input row, and scatters into the streams.
    pub(super) fn of(
        decoder: &magnitude_family_contracts::Decoder,
        source: Option<crate::operators::SublayerSource<'_>>,
        label: &str,
    ) -> Result<Self, CatalogFailure> {
        match source {
            Some(
                source @ (crate::operators::SublayerSource::Mix(_)
                | crate::operators::SublayerSource::Sinkhorn(_)),
            ) => {
                let geometry = crate::operators::streams::StreamsGeometry::of(decoder)
                    .map_err(|error| qualification_dynamic("target", label, error))?
                    .ok_or_else(|| {
                        qualification_dynamic("target", label, "a mix site without streams")
                    })?;
                Ok(Self {
                    form: source.form(geometry.streams),
                    streams: geometry.streams,
                    width: geometry.join_width(),
                })
            }
            _ => Ok(Self::SINGLE),
        }
    }

    /// The width of a residual row of `hidden` streams.
    pub(super) fn residual_width(&self, hidden: u64) -> u64 {
        self.streams * hidden
    }

    /// `rows` input rows the consumer entry forms its input from: `PT`
    /// parts of `hidden`, each of ones.
    pub(super) fn input(
        &self,
        device: &Device,
        rows: u64,
        hidden: u64,
        entry: &'static str,
        label: &str,
    ) -> Result<Tensor, CatalogFailure> {
        semantic_ones(
            device,
            Element::f32(),
            &[rows, self.form.parts * hidden],
            entry,
            label,
        )
    }

    /// The consumer's zero input norm of `element`, empty when the form
    /// normalizes nothing.
    pub(super) fn norm(
        &self,
        device: &Device,
        element: Element,
        hidden: u64,
        entry: &'static str,
        label: &str,
    ) -> Result<Tensor, CatalogFailure> {
        semantic_zeros(
            device,
            element,
            &[self.form.norm_extent(hidden)],
            entry,
            label,
        )
    }

    /// The output entry's join values of `rows` rows: one inject value per
    /// stream (of ones), and of a Sinkhorn site the identity comb after
    /// them, which leaves each stream its own; or none.
    pub(super) fn inject(
        &self,
        device: &Device,
        rows: u64,
        entry: &'static str,
        label: &str,
    ) -> Result<Tensor, CatalogFailure> {
        let s = self.streams;
        let row = (0..self.width).map(|at| match at.checked_sub(s) {
            None => 1.0,
            Some(comb) => f32::from(u8::from(comb / s == comb % s)),
        });
        let values = (0..rows).flat_map(|_| row.clone()).collect::<Vec<_>>();
        semantic_dense_values(device, Element::f32(), &[rows, self.width], &values, entry, label)
    }
}

/// One attention block in its binding's form (the kernels fix it
/// statically) on one row, reading and joining `rows`: zero weights, a zero
/// history row and the row's own fresh key, through both the decode and the
/// prefill entry. Zero weights make each attention output zero, so the block
/// returns the residual.
pub(super) fn qualify_attention(
    device: &Device,
    kernels: &AttentionKernels,
    binding: AttentionBinding,
    rows: SublayerRows,
    label: &str,
) -> Result<(), CatalogFailure> {
    const ENTRY: &str = "attention";
    let shape = binding.shape;
    let (hidden, kv_heads, width, heads) =
        (shape.widths.hidden, shape.widths.kv_heads, shape.widths.width, shape.heads());
    let residual_values = (0..rows.residual_width(hidden))
        .map(|index| (index % 7) as f32 - 3.0)
        .collect::<Vec<_>>();
    let residual = semantic_f32(
        device,
        &[1, rows.residual_width(hidden)],
        &residual_values,
        ENTRY,
        label,
    )?;
    let input = rows.input(device, 1, hidden, ENTRY, label)?;
    let inject = rows.inject(device, 1, "attention_output", label)?;
    let zeros = |element, extents: &[u64]| semantic_zeros(device, element, extents, ENTRY, label);
    let input_norm = rows.norm(device, binding.norm, hidden, ENTRY, label)?;
    // The query segment: the query rows, or a low-rank query's `qr` rows.
    let query = zeros(binding.query_segment(), &[binding.query_segment_rows(), hidden])?;
    let gate = zeros(binding.gate, &[shape.gate_rows(), hidden])?;
    let key = zeros(binding.key, &[shape.key_rows(), hidden])?;
    let value = zeros(binding.value, &[shape.value_rows(), hidden])?;
    let (output_q, output_w) = binding.output_input();
    let output = zeros(binding.output, &[hidden, output_q * output_w])?;
    // A grouped output's zero `a` (G blocks of `rank` rows over `inputs`).
    let grouped = binding
        .grouped
        .map(|grouped| {
            let shape = grouped.shape;
            zeros(grouped.element, &[shape.groups * shape.rank, shape.inputs])
        })
        .transpose()?;
    let query_norm = zeros(Element::f32(), &[u64::from(shape.form.query_norm), width])?;
    let key_norm = zeros(Element::f32(), &[u64::from(shape.form.key_norm), width])?;
    let value_norm = zeros(Element::f32(), &[u64::from(shape.form.value_norm), width])?;
    // The sink logits (zero rows without sinks).
    let sinks = zeros(Element::f32(), &[u64::from(shape.form.sinks), heads])?;
    let pairs = [shape.widths.rotary_pairs];
    let rotary_components = zeros(Element::i32(), &pairs)?;
    let rotary_frequencies = zeros(Element::f32(), &pairs)?;
    let rotary_amplitudes = zeros(Element::f32(), &pairs)?;
    let coordinates = zeros(Element::i32(), &[1, 4])?;
    let visible = semantic_i32(device, &[1, 1, 2], &[0, 1], ENTRY, label)?;
    let fresh = semantic_i32(device, &[1, 2], &[0, 1], ENTRY, label)?;
    let destinations = semantic_i32(device, &[1], &[1], ENTRY, label)?;
    // The affine prefill kernel of a launch that lists no history row tiles.
    let history_tiles = semantic_i32(device, &[0, 1], &[], ENTRY, label)?;
    // The history's prefill reads every visible row: no WorkList bitmap, its
    // blocks of one key.
    let block_slots = semantic_i32(device, &[0, 1], &[], ENTRY, label)?;
    // The rotary layout, output rotary and stored-key values, carried by an
    // empty table's extents.
    let form = semantic_i32(
        device,
        &[
            0,
            u64::from(shape.form.rotary_adjacent),
            u64::from(shape.form.rotary_trailing),
            u64::from(shape.form.output_rotary),
            shape.stored_key(),
        ],
        &[],
        ENTRY,
        label,
    )?;
    // The dense entries' form also carries the key rounding (`KR`), which
    // the check runs at the layers' own rounding, and their index-list
    // source takes no picks (`[1, KS]`, each -1) of no plane rows.
    let rounding = shape.key_rounding;
    let dense_form = semantic_i32(
        device,
        &[
            0,
            u64::from(shape.form.rotary_adjacent),
            u64::from(shape.form.rotary_trailing),
            u64::from(shape.form.output_rotary),
            shape.stored_key(),
            shape.rounds_keys(),
            u64::from(shape.form.absorbed_query),
        ],
        &[],
        ENTRY,
        label,
    )?;
    let selected_rows = semantic_i32(
        device,
        &[1, shape.widths.picks],
        &vec![-1; shape.widths.picks as usize],
        ENTRY,
        label,
    )?;
    let selected = zeros(binding.activation, &[0, shape.widths.kv_heads, shape.widths.width])?;
    let selection = Tensor::from_host(device, Element::u32(), &[1, 0], &[])
        .map_err(|error| qualification_dynamic(ENTRY, label, error))?;
    let scale = |name| unit_scale(device, &kernels.project, name, ENTRY, label);
    let projected = kernels
        .project
        .call(attention_project::Args {
            hidden: &input,
            input_norm: &input_norm,
            query_weight: &query,
            gate_weight: &gate,
            key_weight: &key,
            value_weight: &value,
            epsilon: 1.0e-5,
            project_mode: 0,
            query_scale: &scale("QS")?,
            gate_scale: &scale("GS")?,
            key_scale: &scale("KS")?,
            value_scale: &scale("VS")?,
            query_bound: &any_row_bound(device, &kernels.project, "QB", ENTRY, label)?,
            gate_bound: &any_row_bound(device, &kernels.project, "GB", ENTRY, label)?,
            key_bound: &any_row_bound(device, &kernels.project, "KB", ENTRY, label)?,
            value_bound: &any_row_bound(device, &kernels.project, "VB", ENTRY, label)?,
        })
        .map_err(|e| qualification_dynamic("attention_project", label, e))?;
    let per_head = |tensor: &Tensor, extents: &[u64]| {
        tensor
            .reshape(extents)
            .map_err(|error| qualification_dynamic(ENTRY, label, error))
    };
    // A low-rank query's `qr` row, widened, normalized and projected to the
    // query heads.
    let query_rows = match (binding.low_rank_query, &kernels.low_rank_query) {
        (None, None) => projected.r0,
        (Some(low_rank), Some(low_rank_kernels)) => {
            let qr = low_rank_kernels
                .widen
                .call(widen_rows::Args { rows: &projected.r0 })
                .map_err(|e| qualification_dynamic("widen_rows", label, e))?
                .value;
            let up = zeros(binding.query, &[binding.query_rows(), low_rank.rank])?;
            let empty = zeros(binding.query, &[0, low_rank.rank])?;
            let kernel = &low_rank_kernels.project;
            let scale = |name| unit_scale(device, kernel, name, ENTRY, label);
            let bound = |name| any_row_bound(device, kernel, name, ENTRY, label);
            kernel
                .call(attention_project::Args {
                    hidden: &qr,
                    input_norm: &zeros(low_rank.norm, &[low_rank.rank])?,
                    query_weight: &up,
                    gate_weight: &empty,
                    key_weight: &empty,
                    value_weight: &empty,
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
                .map_err(|e| qualification_dynamic("attention_project", label, e))?
                .r0
        }
        (low_rank, low_rank_kernels) => unreachable!(
            "preparation gives a low-rank query, and only one, its kernels: {low_rank:?} with \
             kernels {}",
            low_rank_kernels.is_some()
        ),
    };
    // An absorbed query's heads, each through its zero block of the key
    // projections.
    let query_rows = match (binding.absorbed, &kernels.absorbed) {
        (None, None) => query_rows,
        (Some(absorbed), Some(kernel)) => {
            let shape = absorbed.shape;
            kernel
                .call(project_rows::Args {
                    source: &per_head(&query_rows, &[1, shape.heads * shape.width])?,
                    weight: &zeros(absorbed.element, &[shape.heads * shape.keys, shape.width])?,
                    weight_scale: &unit_scale(device, kernel, "WS", "project_rows", label)?,
                    weight_bound: &any_row_bound(device, kernel, "WB", "project_rows", label)?,
                })
                .map_err(|e| qualification_dynamic("project_rows", label, e))?
                .value
        }
        (absorbed, kernel) => unreachable!(
            "preparation gives an absorbed query, and only one, its kernel: {absorbed:?} with \
             kernel {}",
            kernel.is_some()
        ),
    };
    let mixed_query = per_head(&query_rows, &[1, heads, width + shape.interleaved_gate()])?;
    let mixed_gate = per_head(&projected.r1, &[1, heads, shape.separate_gate()])?;
    let fresh_rows = [u64::from(shape.form.fresh), 1, kv_heads * width];
    let mixed_key = per_head(&projected.r2, &fresh_rows)?;
    let mixed_value = match shape.form.values {
        AttentionValues::Projected => per_head(&projected.r3, &fresh_rows)?,
        AttentionValues::RawKey => per_head(&projected.r2, &fresh_rows)?,
        // Stored-key values take no value of their own (`VK`).
        AttentionValues::StoredKey => zeros(binding.activation, &[0, 1, kv_heads * width])?,
    };
    // Every entry takes the same arguments but its history planes, which
    // start zero (a zero key row and a zero value row). A plane of no rows
    // is a stored-key history's absent value plane.
    let value_rows = if shape.stored_key() == 1 { 0 } else { 2 };
    macro_rules! mix {
        ($kernel:expr, $module:ident, $($plane:ident: $element:expr, $elements:expr, $rows:expr),*) => {
            mix!(@call $kernel, $module, {form: &form}, $($plane: $element, $elements, $rows),*)
        };
        (@call $kernel:expr, $module:ident, {$($extra:ident: $value:expr),*},
            $($plane:ident: $element:expr, $elements:expr, $rows:expr),*) => {{
            let mut slabs = Vec::new();
            $(
                let mut $plane = if $rows > 0 {
                    let mut slab = seismic::SlabTensor::new(device, 2, 2, vec![
                        seismic::SlabRegion {
                            element: $element,
                            row_shape: vec![kv_heads, $elements],
                        },
                    ])
                    .map_err(|error| qualification_dynamic(ENTRY, label, error))?;
                    slab.add_slab()
                        .map_err(|error| qualification_dynamic(ENTRY, label, error))?;
                    let region = slab
                        .logical_region(0)
                        .map_err(|error| qualification_dynamic(ENTRY, label, error))?;
                    slabs.push(slab);
                    region
                } else {
                    zeros($element, &[0, kv_heads, $elements])?
                };
            )*
            $kernel
                .call($module::Args {
                    query: &mixed_query,
                    gate: &mixed_gate,
                    key: &mixed_key,
                    value: &mixed_value,
                    query_norm: &query_norm,
                    key_norm: &key_norm,
                    value_norm: &value_norm,
                    sinks: &sinks,
                    rotary_components: &rotary_components,
                    rotary_frequencies: &rotary_frequencies,
                    rotary_amplitudes: &rotary_amplitudes,
                    coordinates: &coordinates,
                    visible: &visible,
                    fresh: &fresh,
                    destinations: &destinations,
                    slab_rows: 2,
                    $($extra: $value,)*
                    $($plane: &mut $plane,)*
                    epsilon: 1.0e-5,
                    scale: 1.0 / (width as f32).sqrt(),
                    gate_function: 0,
                })
                .map_err(|e| qualification_dynamic(stringify!($module), label, e))?
                .value
        }};
    }
    let activation = binding.activation;
    let mixed = match &kernels.history {
        AttentionHistoryKernels::Dense {
            decode, prefill, ..
        } => [
            match decode {
                DenseDecodeKernels::Decode { single, .. } => (
                    "attention_decode",
                    mix!(@call single, attention_decode, {
                        form: &dense_form, selected_rows: &selected_rows, selected: &selected,
                        selected_slab_rows: 1, key_block: rounding.block, key_element: rounding.element,
                        key_scale: rounding.scale, key_amax_floor: rounding.amax_floor()
                    }, history_key: activation, width, 2, history_value: activation, width, value_rows),
                ),
                DenseDecodeKernels::Prefill(kernel) => (
                    "attention_prefill",
                    mix!(@call kernel, attention_prefill, {
                        selection: &selection, block_slots: &block_slots,
                        form: &dense_form, selected_rows: &selected_rows, selected: &selected,
                        selected_slab_rows: 1, key_block: rounding.block, key_element: rounding.element,
                        key_scale: rounding.scale, key_amax_floor: rounding.amax_floor()
                    }, history_key: activation, width, 2, history_value: activation, width, value_rows),
                ),
            },
            (
                "attention_prefill",
                mix!(@call prefill, attention_prefill, {
                    selection: &selection, block_slots: &block_slots,
                    form: &dense_form, selected_rows: &selected_rows, selected: &selected,
                    selected_slab_rows: 1, key_block: rounding.block, key_element: rounding.element,
                    key_scale: rounding.scale, key_amax_floor: rounding.amax_floor()
                }, history_key: activation, width, 2, history_value: activation, width, value_rows),
            ),
        ],
        AttentionHistoryKernels::AffineK8V4 {
            decode, prefill, ..
        } => {
            let geometry = binding
                .history
                .affine_geometry()
                .map_err(|e| qualification_dynamic(ENTRY, label, e.to_string()))?;
            [
                (
                    "attention_decode_k8v4",
                    mix!(decode, attention_decode_k8v4,
                        history_key_codes: Element::u32(), geometry.key_codes as u64, 2,
                        history_key_coefficients: Element::f16(), geometry.key_coefficients as u64, 2,
                        history_value_codes: Element::u32(), geometry.value_codes as u64, 2,
                        history_value_coefficients: Element::f16(), geometry.value_coefficients as u64, 2),
                ),
                (
                    "attention_prefill_k8v4",
                    mix!(@call prefill, attention_prefill_k8v4,
                        {selection: &selection, block_slots: &block_slots, history_tiles: &history_tiles, form: &form},
                        history_key_codes: Element::u32(), geometry.key_codes as u64, 2,
                        history_key_coefficients: Element::f16(), geometry.key_coefficients as u64, 2,
                        history_value_codes: Element::u32(), geometry.value_codes as u64, 2,
                        history_value_coefficients: Element::f16(), geometry.value_coefficients as u64, 2),
                ),
            ]
        }
    };
    for (entry, gated) in mixed.iter().map(|(entry, gated)| (*entry, gated)) {
        // A grouped output's products are the rows its output projection
        // reads.
        let products = match (&kernels.grouped, &grouped) {
            (None, None) => None,
            (Some(kernel), Some(weight)) => Some(
                kernel
                    .call(project_rows::Args {
                        source: &gated
                            .reshape(&[1, heads * width])
                            .map_err(|e| qualification_dynamic("project_rows", label, e))?,
                        weight,
                        weight_scale: &unit_scale(device, kernel, "WS", "project_rows", label)?,
                        weight_bound: &any_row_bound(device, kernel, "WB", "project_rows", label)?,
                    })
                    .map_err(|e| qualification_dynamic("project_rows", label, e))?
                    .value
                    .reshape(&[1, output_q, output_w])
                    .map_err(|e| qualification_dynamic("project_rows", label, e))?,
            ),
            (kernel, weight) => unreachable!(
                "preparation gives a grouped output, and only one, its kernels: kernels {} with \
                 weight {}",
                kernel.is_some(),
                weight.is_some()
            ),
        };
        let gated = products.as_ref().unwrap_or(gated);
        let result = match (&kernels.output, binding.tail) {
            (SublayerOutput::Residual(output_kernel), SublayerTail::Residual) => {
                output_kernel
                    .call(attention_output::Args {
                        hidden: &residual,
                        inject: &inject,
                        gated,
                        output_weight: &output,
                        output_scale: &unit_scale(device, output_kernel, "OS", ENTRY, label)?,
                        output_bound: &any_row_bound(device, output_kernel, "OB", ENTRY, label)?,
                    })
                    .map_err(|e| qualification_dynamic("attention_output", label, e))?
                    .value
            }
            (SublayerOutput::PostNorm(tail), SublayerTail::PostNorm { norm, .. }) => {
                let source = gated
                    .reshape(&[1, output_q * output_w])
                    .map_err(|e| qualification_dynamic("project_rows", label, e))?;
                qualify_post_norm(device, tail, &residual, &source, &output, norm, label)?
            }
            _ => {
                return Err(qualification_dynamic(
                    "attention_output",
                    label,
                    "attention output entries disagree with the binding's tail",
                ));
            }
        };
        require_f32_values(&result, &residual_values, entry, label)?;
    }
    Ok(())
}

/// A post-norm tail over one row: `source` projected to F32 by zero
/// `weight`, normalized with zero norm weights into `residual`, which it
/// leaves unchanged.
pub(super) fn qualify_post_norm(
    device: &Device,
    kernels: &PostNormKernels,
    residual: &Tensor,
    source: &Tensor,
    weight: &Tensor,
    norm: Element,
    label: &str,
) -> Result<Tensor, CatalogFailure> {
    let hidden = residual.extents()[1];
    let projected = kernels
        .project
        .call(project_rows::Args {
            source,
            weight,
            weight_scale: &unit_scale(device, &kernels.project, "WS", "project_rows", label)?,
            weight_bound: &any_row_bound(device, &kernels.project, "WB", "project_rows", label)?,
        })
        .map_err(|e| qualification_dynamic("project_rows", label, e))?
        .value;
    let norm = semantic_zeros(device, norm, &[hidden], "post_norm_residual", label)?;
    let out_rows = semantic_i32(device, &[1], &[0], "post_norm_residual", label)?;
    Ok(kernels
        .residual
        .call(post_norm_residual::Args {
            residual,
            projected: &projected,
            norm: &norm,
            out_rows: &out_rows,
            epsilon: 1.0e-8,
            scale: 1.0,
        })
        .map_err(|e| qualification_dynamic("post_norm_residual", label, e))?
        .value)
}

pub(super) fn one_bytes(dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => 1.0_f32.to_le_bytes().to_vec(),
        DType::F16 => 0x3c00_u16.to_le_bytes().to_vec(),
        DType::BF16 => 0x3f80_u16.to_le_bytes().to_vec(),
        _ => unreachable!("dense import catalog contains floating dtypes only"),
    }
}

pub(super) fn tensor_f32(
    device: &Device,
    extents: &[u64],
    values: &[f32],
    entry: &'static str,
    bindings: &'static str,
) -> Result<Tensor, CatalogFailure> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::f32(), extents, &bytes)
        .map_err(|error| qualification(entry, bindings, error))
}

/// `rows` (`[M, PT * D]`) as the `[M, PT, D]` parts a readout entry reads,
/// PT the entry's static (1 when it declares none).
pub(super) fn readout_parts(
    rows: &Tensor,
    statics: &std::collections::BTreeMap<String, u64>,
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    let parts = statics.get("PT").copied().unwrap_or(1);
    let [m, width] = rows.extents()[..] else {
        return Err(qualification_dynamic(
            entry,
            bindings,
            "readout rows are not a matrix",
        ));
    };
    rows.reshape(&[m, parts, width / parts])
        .map_err(|error| qualification_dynamic(entry, bindings, error))
}

pub(super) fn semantic_zeros(
    device: &Device,
    element: Element,
    extents: &[u64],
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    Tensor::zeros(device, element, extents)
        .map_err(|error| qualification_dynamic(entry, bindings, error))
}

/// A packing bound for `kernel`'s bound port whose static is `name`: its
/// static extent (0 where the native has no such static) of the largest
/// squared run norm any packable format has (32 centred 8-bit codes of
/// magnitude 128), which bounds whatever weights a qualification binds.
pub(super) fn any_row_bound<E: seismic::Entry>(
    device: &Device,
    kernel: &NativeKernel<E>,
    name: &str,
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    let extent = kernel.specialization().static_value(name).unwrap_or(0);
    let count = usize::try_from(extent).map_err(|error| qualification_dynamic(entry, bindings, error))?;
    semantic_f32(device, &[extent], &vec![32.0 * 128.0 * 128.0; count], entry, bindings)
}

/// A gated expansion's host port `glu_limit` for `limit`: no values, at
/// extents `[0, LB]` (`GluLimit::word`).
pub(super) fn glu_limit(
    device: &Device,
    limit: Option<magnitude_family_contracts::GluLimit>,
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    semantic_i32(device, &[0, magnitude_family_contracts::GluLimit::word(limit)], &[], entry, bindings)
}

/// A unit accumulator scale for `kernel`'s scale port whose static is
/// `name`: ones at its static extent (0, absent, where the native has no
/// such static), which leave whatever a qualification binds unscaled.
pub(super) fn unit_scale<E: seismic::Entry>(
    device: &Device,
    kernel: &NativeKernel<E>,
    name: &str,
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    let extent = kernel.specialization().static_value(name).unwrap_or(0);
    semantic_ones(device, Element::f32(), &[extent], entry, bindings)
}

pub(super) fn semantic_f32(
    device: &Device,
    extents: &[u64],
    values: &[f32],
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::f32(), extents, &bytes)
        .map_err(|error| qualification_dynamic(entry, bindings, error))
}

pub(super) fn semantic_i32(
    device: &Device,
    extents: &[u64],
    values: &[i32],
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::i32(), extents, &bytes)
        .map_err(|error| qualification_dynamic(entry, bindings, error))
}

pub(super) fn semantic_dense_values(
    device: &Device,
    element: Element,
    extents: &[u64],
    values: &[f32],
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    let bytes: Vec<u8> = match element.dtype() {
        Some(DType::F32) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        Some(DType::F16) => values
            .iter()
            .flat_map(|value| seismic::f16_bits(*value).to_le_bytes())
            .collect(),
        Some(DType::BF16) => values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        _ => {
            return Err(qualification_dynamic(
                entry,
                bindings,
                format!("{} is not a dense floating representation", element.name()),
            ));
        }
    };
    Tensor::from_host(device, element, extents, &bytes)
        .map_err(|error| qualification_dynamic(entry, bindings, error))
}

pub(super) fn semantic_ones(
    device: &Device,
    element: Element,
    extents: &[u64],
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    let count = crate::planning::elements(extents)
        .ok_or_else(|| qualification_dynamic(entry, bindings, "fixture extent overflow"))?;
    let count = usize::try_from(count)
        .map_err(|_| qualification_dynamic(entry, bindings, "fixture extent exceeds usize"))?;
    semantic_dense_values(device, element, extents, &vec![1.0; count], entry, bindings)
}

pub(super) fn semantic_pattern(
    device: &Device,
    element: Element,
    extents: &[u64],
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogFailure> {
    if element.dtype().is_some() {
        return semantic_ones(device, element, extents, entry, bindings);
    }
    let (source, packet) = pattern_source(element.representation()).ok_or_else(|| {
        qualification_dynamic(
            entry,
            bindings,
            format!(
                "no non-degenerate fixture exists for {}",
                element.representation()
            ),
        )
    })?;
    let repack = |extents: &[u64]| {
        pattern_bytes(element, source, &packet, extents).ok_or_else(|| {
            qualification_dynamic(entry, bindings, "no registered fixture conversion")
        })
    };
    let mut tensor = semantic_zeros(device, element, extents, entry, bindings)?;
    let storage = usize::try_from(tensor.storage_bytes())
        .map_err(|_| qualification_dynamic(entry, bindings, "fixture storage exceeds usize"))?;
    // Every source packet is identical and packed layouts place whole row
    // groups contiguously, so a large matrix is one converted row block
    // repeated. The host reference converts bit by bit; converting a
    // vocabulary-sized projection whole costs seconds at every load.
    let tiled = match extents {
        [rows, columns] if *rows > PATTERN_BLOCK_ROWS && rows % PATTERN_BLOCK_ROWS == 0 => {
            let block = repack(&[PATTERN_BLOCK_ROWS, *columns])?;
            let repeats = usize::try_from(rows / PATTERN_BLOCK_ROWS)
                .map_err(|_| qualification_dynamic(entry, bindings, "fixture exceeds usize"))?;
            (block.len().checked_mul(repeats) == Some(storage)).then(|| block.repeat(repeats))
        }
        _ => None,
    };
    let pattern = match tiled {
        Some(pattern) => pattern,
        None => repack(extents)?,
    };
    tensor
        .write_from_host(&pattern)
        .map_err(|error| qualification_dynamic(entry, bindings, error))?;
    Ok(tensor)
}

/// Rows converted once for a repeated fixture matrix: a multiple of every
/// packed layout's row group.
const PATTERN_BLOCK_ROWS: u64 = 256;

/// One GGUF source packet whose every value decodes to 1.0, for a packed
/// representation. Repeated and converted by the registry's host reference
/// into the element's (representation, layout), the fixture follows every
/// layout's geometry.
fn pattern_source(representation: &str) -> Option<(Element, Vec<u8>)> {
    let source = gguf_source(representation)?;
    let size = usize::try_from(
        source
            .canonical_byte_len(&[1, source.logical_group()?])
            .ok()?,
    )
    .ok()?;
    let packet = match representation {
        "q8g32s" => unit_packet(size, &[(0, &[0x00, 0x3c]), (2, &[1; 32])]),
        // Codes 17 (low nibble 1, high bit 1) at offset -16 and d = 1.
        "q5g32s" => unit_packet(
            size,
            &[(0, &[0x00, 0x3c]), (2, &[0xff; 4]), (6, &[0x11; 16])],
        ),
        // d = 1, dmin = 0, sub-block scales 1 and minima 0, codes 1.
        "q4k" => unit_packet(
            size,
            &[
                (0, &[0x00, 0x3c]),
                (4, &[1; 4]),
                (12, &[1; 4]),
                (16, &[0x11; 128]),
            ],
        ),
        "q5k" => unit_packet(
            size,
            &[
                (0, &[0x00, 0x3c]),
                (4, &[1; 4]),
                (12, &[1; 4]),
                (48, &[0x11; 128]),
            ],
        ),
        // Codes 33 (low nibble 1, high bits 2) at scale 1 and d = 1.
        "q6k" => unit_packet(
            size,
            &[
                (0, &[0x11; 128]),
                (128, &[0xaa; 64]),
                (192, &[1; 16]),
                (208, &[0x00, 0x3c]),
            ],
        ),
        // Codes 9 at offset -8 and d = 1.
        "q4g32s" => unit_packet(size, &[(0, &[0x00, 0x3c]), (2, &[0x99; 16])]),
        // Codes 1, d = 1 and minimum m = 0.
        "q4g32" => unit_packet(size, &[(0, &[0x00, 0x3c]), (4, &[0x11; 16])]),
        // Codes 1, d = 1 and minimum m = 0.
        "q5g32" => unit_packet(size, &[(0, &[0x00, 0x3c]), (8, &[0x11; 16])]),
        // E2M1 code 2 (value 1) and E8M0 scale 127 (2^0).
        "mxfp4g32" => unit_packet(size, &[(0, &[127]), (1, &[0x22; 16])]),
        // A planar block (`pattern_bytes`): E4M3 codes 0x38 (value 1), then
        // their E8M0 scale 127 (2^0).
        "mxfp8g32" => unit_packet(size, &[(0, &[0x38; 32]), (32, &[127])]),
        // A planar block: E4M3 codes 0x38 (value 1), then their F32 scale 1.0.
        "fp8g128" => unit_packet(size, &[(0, &[0x38; 128]), (128, &1.0f32.to_le_bytes())]),
        // E2M1 code 2 (value 1) and four UE4M3 scales 0x38 (1.0).
        "nvfp4g16" => unit_packet(size, &[(0, &[0x38; 4]), (4, &[0x22; 32])]),
        // Grid point 0 (every magnitude 8), no signs, scale codes 0 (odd
        // scale 1) and d = 1: d / 8 * 1 * 8.
        "iq2xs" => unit_packet(size, &[(0, &[0x00, 0x3c])]),
        // Grid points 0 (every magnitude 4), no signs, scale codes 0 and
        // d = 1: d / 4 * 1 * 4.
        "iq3xxs" => unit_packet(size, &[(0, &[0x00, 0x3c])]),
        // Grid point 0 (every magnitude 8), no signs, scale codes 0 and
        // d = 1: d / 8 * 1 * 8.
        "iq2xxs" => unit_packet(size, &[(0, &[0x00, 0x3c])]),
        // Sub-block scales 1 and minima 0, codes 1, d = 1 and dmin = 0.
        "q2k" => unit_packet(
            size,
            &[(0, &[0x01; 16]), (16, &[0x55; 64]), (80, &[0x00, 0x3c])],
        ),
        // Codes 2 (value 2 - 1) and d = 1.
        "q2g256" => unit_packet(size, &[(0, &[0xaa; 64]), (64, &[0x00, 0x3c])]),
        // d = 1 and codes 2 (value 2 - 1).
        "q2g64" => unit_packet(size, &[(0, &[0x00, 0x3c]), (2, &[0xaa; 16])]),
        // d = 1 and every bit set: `+d`.
        "q1g128" => unit_packet(size, &[(0, &[0x00, 0x3c]), (2, &[0xff; 16])]),
        // Grid point 0 (every magnitude 8), no signs, scale codes 0 and
        // d = 1: d / 8 * 1 * 8.
        "iq2s" => unit_packet(size, &[(0, &[0x00, 0x3c])]),
        // Grid point 1029 (every value 0), scale codes 0, the bias +1/8 and
        // d = 8: 8 * 1 * (0 + 1/8).
        "iq1s" => unit_packet(
            size,
            &[
                (0, &[0x00, 0x48]),
                (2, &[0x05; 32]),
                (34, &[0x24, 0x09].repeat(8)[..]),
            ],
        ),
        // Grid point 1029 (every value 0), deltas +1, scale codes 0 and
        // d = 8 (its nibbles the scales' top nibbles): 8 / 8 * 1 * (8 0 + 1).
        "iq1m" => unit_packet(
            size,
            &[
                (0, &[0x05; 32]),
                (32, &[0x44; 16]),
                (52, &[0x00, 0x80, 0x00, 0x40]),
            ],
        ),
        // Table code 8 (value 1) at sub-scale 33 - 32 = 1 and d = 1.
        "iq4xs" => unit_packet(
            size,
            &[
                (0, &[0x00, 0x3c]),
                (2, &[0xaa, 0xaa]),
                (4, &[0x11; 4]),
                (8, &[0x88; 128]),
            ],
        ),
        // Table code 8 (value 1) and d = 1.
        "iq4g32s" => unit_packet(size, &[(0, &[0x00, 0x3c]), (2, &[0x88; 16])]),
        // Codes 5 (low bits 1, high bit 1: value 5 - 4) at sub-scale
        // 33 - 32 = 1 (low nibbles 1, high bits 2) and d = 1.
        "q3k" => unit_packet(
            size,
            &[
                (0, &[0xff; 32]),
                (32, &[0x55; 64]),
                (96, &[0x11; 8]),
                (104, &[0xaa; 4]),
                (108, &[0x00, 0x3c]),
            ],
        ),
        // Grid point 0 (every magnitude 1), no signs, scale codes 0 (odd
        // scale 1) and d = 1.
        "iq3s" => unit_packet(size, &[(0, &[0x00, 0x3c])]),
        _ => return None,
    };
    Some((source, packet))
}

/// The registered GGUF source a packed representation is converted from.
fn gguf_source(representation: &str) -> Option<Element> {
    Element::external_sources().find(|source| {
        source
            .converted(Layout::Packet, OperandRecipe::Rounded)
            .is_some_and(|packet| packet.representation() == representation)
    })
}

/// The host-converted fixture of `extents`: `packet` repeated over the
/// source's canonical bytes and converted into `element`. A planar source's
/// packet is its codes then its scale (`Element::record_parts` of one row of
/// a group): its codes plane repeats the codes, its grid the scale.
fn pattern_bytes(
    element: Element,
    source: Element,
    packet: &[u8],
    extents: &[u64],
) -> Option<Vec<u8>> {
    let parts = source.record_parts(extents)?;
    let packet_parts = source.record_parts(&[1, source.logical_group()?])?;
    let mut at = 0;
    let mut source_bytes = Vec::new();
    for (part, packet_part) in parts.into_iter().zip(packet_parts) {
        let piece = &packet[at..at + usize::try_from(packet_part).ok()?];
        at += piece.len();
        source_bytes.extend(piece.iter().copied().cycle().take(usize::try_from(part).ok()?));
    }
    element.repack_host(source, extents, &source_bytes)
}

/// A source packet of `size` bytes: zero except the given byte runs.
fn unit_packet(size: usize, runs: &[(usize, &[u8])]) -> Vec<u8> {
    let mut packet = vec![0_u8; size];
    for (offset, bytes) in runs {
        packet[*offset..*offset + bytes.len()].copy_from_slice(bytes);
    }
    packet
}

pub(super) fn require_zero_result(
    tensor: &Tensor,
    entry: &'static str,
    bindings: &str,
) -> Result<(), CatalogFailure> {
    let bytes = tensor
        .read_to_host()
        .map_err(|error| qualification_dynamic(entry, bindings, error))?;
    if bytes.iter().any(|byte| *byte != 0) {
        return Err(qualification_dynamic(
            entry,
            bindings,
            "zero-input semantic fixture produced a nonzero result",
        ));
    }
    Ok(())
}

pub(super) fn require_f32_values(
    tensor: &Tensor,
    expected: &[f32],
    entry: &'static str,
    bindings: &str,
) -> Result<(), CatalogFailure> {
    let bytes = tensor
        .read_to_host()
        .map_err(|error| qualification_dynamic(entry, bindings, error))?;
    let actual = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if actual != expected {
        return Err(qualification_dynamic(
            entry,
            bindings,
            "semantic fixture result mismatch",
        ));
    }
    Ok(())
}

pub(super) fn require_finite_nonzero_f32(
    tensor: &Tensor,
    entry: &'static str,
    bindings: &str,
) -> Result<(), CatalogFailure> {
    let bytes = tensor
        .read_to_host()
        .map_err(|error| qualification_dynamic(entry, bindings, error))?;
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if values.is_empty()
        || values.iter().any(|value| !value.is_finite())
        || values.iter().all(|value| *value == 0.0)
    {
        return Err(qualification_dynamic(
            entry,
            bindings,
            "semantic fixture was not finite and nonzero",
        ));
    }
    Ok(())
}

pub(super) fn require_finite_nonzero(
    tensor: &Tensor,
    entry: &'static str,
    bindings: &str,
) -> Result<(), CatalogFailure> {
    let values = read_dense_values(tensor, entry, bindings)?;
    if values.is_empty()
        || values.iter().any(|value| !value.is_finite())
        || values.iter().all(|value| *value == 0.0)
    {
        return Err(qualification_dynamic(
            entry,
            bindings,
            "semantic fixture was not finite and nonzero",
        ));
    }
    Ok(())
}

pub(super) fn read_dense_values(
    tensor: &Tensor,
    entry: &'static str,
    bindings: &str,
) -> Result<Vec<f32>, CatalogFailure> {
    let bytes = tensor
        .read_to_host()
        .map_err(|error| qualification_dynamic(entry, bindings, error))?;
    match tensor.element().dtype() {
        Some(DType::F32) => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect()),
        Some(DType::F16) => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| {
                seismic::f16_to_f32(u16::from_le_bytes(
                    chunk.try_into().expect("two-byte chunk"),
                ))
            })
            .collect()),
        Some(DType::BF16) => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| {
                f32::from_bits(
                    u32::from(u16::from_le_bytes(
                        chunk.try_into().expect("two-byte chunk"),
                    )) << 16,
                )
            })
            .collect()),
        _ => Err(qualification_dynamic(
            entry,
            bindings,
            "result is not dense floating point",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every packed representation with a GGUF source, under every operand
    /// recipe: the elements a fixture is formed for.
    fn gguf_elements(layout: Layout) -> Vec<Element> {
        Element::external_sources()
            .flat_map(|source| {
                OperandRecipe::ALL
                    .into_iter()
                    .filter_map(move |operands| source.converted(layout, operands))
            })
            .collect()
    }

    /// Columns of whole blocks of `source` with a partial last 256-value
    /// block where the source's blocks are smaller: three 256-value blocks
    /// less one source block.
    fn columns(source: Element) -> u64 {
        3 * 256 - source.logical_group().unwrap()
    }

    /// A large fixture is converted as one repeated row block; the block must
    /// reproduce the whole conversion byte for byte in every layout a GGUF
    /// format converts into.
    #[test]
    fn repeated_pattern_block_matches_the_whole_conversion() {
        for layout in Layout::ALL {
            // A conversion writes one storage for every operand recipe.
            for element in gguf_elements(layout)
                .into_iter()
                .filter(|element| element.operand_recipe() == OperandRecipe::Rounded)
            {
                let representation = element.representation();
                let (source, packet) = pattern_source(representation)
                    .unwrap_or_else(|| panic!("{representation} has no fixture"));
                let columns = columns(source);
                let whole =
                    pattern_bytes(element, source, &packet, &[2 * PATTERN_BLOCK_ROWS, columns])
                        .unwrap();
                let block = pattern_bytes(element, source, &packet, &[PATTERN_BLOCK_ROWS, columns])
                    .unwrap();
                assert_eq!(block.repeat(2), whole, "{representation} {layout:?}");
            }
        }
    }

    /// Every packed representation with a GGUF source has a fixture, and it
    /// decodes to 1.0 everywhere in every layout (at a K with a partial
    /// 256-value block where the source's blocks allow one).
    #[test]
    fn every_packed_representation_has_a_unit_fixture() {
        for layout in Layout::ALL {
            for element in gguf_elements(layout) {
                let representation = element.representation();
                let (source, packet) = pattern_source(representation)
                    .unwrap_or_else(|| panic!("{representation} has no fixture"));
                let extents = [3, columns(source)];
                let bytes = pattern_bytes(element, source, &packet, &extents).unwrap();
                let values = element.decode_host(&extents, &bytes).unwrap();
                assert!(
                    values.iter().all(|value| *value == 1.0),
                    "{} {layout:?}",
                    element.name()
                );
            }
        }
    }
}
