use super::super::*;

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

/// Elements of one attention block's weights.
pub(super) struct AttentionElements {
    pub input_norm: Element,
    pub query_gate: Element,
    pub key: Element,
    pub value: Element,
    pub output: Element,
    pub activation: Element,
}

/// One attention block at the model's attention geometry (the kernels fix it
/// statically) on one row: zero weights, a zero history row and the row's own
/// fresh key, through both the decode and the prefill entry. Zero weights
/// make each attention output zero, so the block returns the residual.
pub(super) fn qualify_attention(
    device: &Device,
    kernels: &AttentionKernels,
    shape: AttentionShape,
    elements: &AttentionElements,
    label: &str,
) -> Result<(), CatalogFailure> {
    const ENTRY: &str = "attention";
    let AttentionShape {
        hidden,
        kv_heads,
        group,
        rotary_pairs,
        width,
    } = shape;
    let heads = kv_heads * group;
    let residual_values = (0..hidden)
        .map(|index| (index % 7) as f32 - 3.0)
        .collect::<Vec<_>>();
    let residual = semantic_f32(device, &[1, hidden], &residual_values, ENTRY, label)?;
    let input_norm = semantic_zeros(device, elements.input_norm, &[hidden], ENTRY, label)?;
    let query_gate = semantic_zeros(
        device,
        elements.query_gate,
        &[heads * 2 * width, hidden],
        ENTRY,
        label,
    )?;
    let key = semantic_zeros(
        device,
        elements.key,
        &[kv_heads * width, hidden],
        ENTRY,
        label,
    )?;
    let value = semantic_zeros(
        device,
        elements.value,
        &[kv_heads * width, hidden],
        ENTRY,
        label,
    )?;
    let output = semantic_zeros(
        device,
        elements.output,
        &[hidden, heads * width],
        ENTRY,
        label,
    )?;
    let query_norm = semantic_zeros(device, Element::f32(), &[width], ENTRY, label)?;
    let key_norm = semantic_zeros(device, Element::f32(), &[width], ENTRY, label)?;
    let rotary_components = semantic_zeros(device, Element::i32(), &[rotary_pairs], ENTRY, label)?;
    let rotary_frequencies = semantic_zeros(device, Element::f32(), &[rotary_pairs], ENTRY, label)?;
    let coordinates = semantic_zeros(device, Element::i32(), &[1, 4], ENTRY, label)?;
    let visible = semantic_i32(device, &[1, 1, 2], &[0, 1], ENTRY, label)?;
    let fresh = semantic_i32(device, &[1, 2], &[0, 1], ENTRY, label)?;
    let destinations = semantic_i32(device, &[1], &[1], ENTRY, label)?;
    let projected = kernels
        .project
        .call(gated_attention_project::Args {
            hidden: &residual,
            input_norm: &input_norm,
            query_norm: &query_norm,
            query_gate_weight: &query_gate,
            key_weight: &key,
            value_weight: &value,
            epsilon: 1.0e-5,
        })
        .map_err(|e| qualification_dynamic("gated_attention_project", label, e))?;
    let plane = |element: Element, elements: u64| {
        semantic_zeros(device, element, &[2, kv_heads, elements], ENTRY, label)
    };
    // Every entry takes the same arguments but its history planes, which
    // start zero (a zero key row and a zero value row).
    macro_rules! mix {
        ($kernel:expr, $module:ident, $($plane:ident: $element:expr, $elements:expr),*) => {{
            $(let mut $plane = plane($element, $elements)?;)*
            $kernel
                .call($module::Args {
                    query_gate: &projected.r0,
                    key: &projected.r1,
                    value: &projected.r2,
                    query_norm: &query_norm,
                    key_norm: &key_norm,
                    rotary_components: &rotary_components,
                    rotary_frequencies: &rotary_frequencies,
                    coordinates: &coordinates,
                    visible: &visible,
                    fresh: &fresh,
                    destinations: &destinations,
                    $($plane: &mut $plane,)*
                    epsilon: 1.0e-5,
                    scale: 1.0 / (width as f32).sqrt(),
                })
                .map_err(|e| qualification_dynamic(stringify!($module), label, e))?
                .value
        }};
    }
    let activation = elements.activation;
    let mixed = match &kernels.history {
        AttentionHistoryKernels::Dense { decode, prefill } => [
            (
                "gated_attention_decode",
                mix!(decode, gated_attention_decode,
                    history_key: activation, width, history_value: activation, width),
            ),
            (
                "gated_attention_prefill",
                mix!(prefill, gated_attention_prefill,
                    history_key: activation, width, history_value: activation, width),
            ),
        ],
        AttentionHistoryKernels::AffineK8V4 { decode, prefill } => {
            let pairs = crate::programs::graph::attention::affine_coefficients(width);
            [
                (
                    "gated_attention_decode_k8v4",
                    mix!(decode, gated_attention_decode_k8v4,
                        history_key_codes: Element::u32(), width / 4,
                        history_key_coefficients: Element::f16(), pairs,
                        history_value_codes: Element::u32(), width / 8,
                        history_value_coefficients: Element::f16(), pairs),
                ),
                (
                    "gated_attention_prefill_k8v4",
                    mix!(prefill, gated_attention_prefill_k8v4,
                        history_key_codes: Element::u32(), width / 4,
                        history_key_coefficients: Element::f16(), pairs,
                        history_value_codes: Element::u32(), width / 8,
                        history_value_coefficients: Element::f16(), pairs),
                ),
            ]
        }
    };
    for (entry, gated) in mixed.iter().map(|(entry, gated)| (*entry, gated)) {
        let result = kernels
            .output
            .call(attention_output::Args {
                hidden: &residual,
                gated,
                output_weight: &output,
            })
            .map_err(|e| qualification_dynamic("attention_output", label, e))?
            .value;
        require_f32_values(&result, &residual_values, entry, label)?;
    }
    Ok(())
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
            .flat_map(|value| f16_bits(*value).to_le_bytes())
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
    let count = extents
        .iter()
        .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
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
    let (source, packet) = match representation {
        "q8g32s" => (
            "gguf_q8_0",
            unit_packet(34, &[(0, &[0x00, 0x3c]), (2, &[1; 32])]),
        ),
        // d = 1, dmin = 0, sub-block scales 1 and minima 0, codes 1.
        "q4k" => (
            "gguf_q4_k",
            unit_packet(
                144,
                &[
                    (0, &[0x00, 0x3c]),
                    (4, &[1; 4]),
                    (12, &[1; 4]),
                    (16, &[0x11; 128]),
                ],
            ),
        ),
        "q5k" => (
            "gguf_q5_k",
            unit_packet(
                176,
                &[
                    (0, &[0x00, 0x3c]),
                    (4, &[1; 4]),
                    (12, &[1; 4]),
                    (48, &[0x11; 128]),
                ],
            ),
        ),
        // Codes 33 (low nibble 1, high bits 2) at scale 1 and d = 1.
        "q6k" => (
            "gguf_q6_k",
            unit_packet(
                210,
                &[
                    (0, &[0x11; 128]),
                    (128, &[0xaa; 64]),
                    (192, &[1; 16]),
                    (208, &[0x00, 0x3c]),
                ],
            ),
        ),
        // Table code 8 (value 1) at sub-scale 33 - 32 = 1 and d = 1.
        "iq4g32" => (
            "gguf_iq4_xs",
            unit_packet(
                136,
                &[
                    (0, &[0x00, 0x3c]),
                    (2, &[0xaa, 0xaa]),
                    (4, &[0x11; 4]),
                    (8, &[0x88; 128]),
                ],
            ),
        ),
        _ => return None,
    };
    Some((
        Element::named(source).expect("registered GGUF source representation"),
        packet,
    ))
}

/// The host-converted fixture of `extents`: `packet` repeated over the
/// source's canonical bytes and converted into `element`.
fn pattern_bytes(
    element: Element,
    source: Element,
    packet: &[u8],
    extents: &[u64],
) -> Option<Vec<u8>> {
    let length = usize::try_from(source.canonical_byte_len(extents).ok()?).ok()?;
    let source_bytes = packet
        .iter()
        .copied()
        .cycle()
        .take(length)
        .collect::<Vec<_>>();
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
                f16_to_f32(u16::from_le_bytes(
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

pub(super) fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        return sign | (((mantissa | 0x80_0000) >> (1 - exponent + 13)) as u16);
    }
    let rounded = mantissa + 0x0fff + ((mantissa >> 13) & 1);
    sign | ((exponent as u16) << 10) | ((rounded >> 13) as u16)
}

pub(super) fn f16_to_f32(value: u16) -> f32 {
    let sign = (u32::from(value & 0x8000)) << 16;
    let exponent = u32::from((value >> 10) & 0x1f);
    let mantissa = u32::from(value & 0x03ff);
    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let shift = mantissa.leading_zeros() - 21;
            sign | ((127 - 15 - shift + 1) << 23) | ((mantissa << (shift + 1) & 0x03ff) << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::Layout;

    /// A large fixture is converted as one repeated row block; the block must
    /// reproduce the whole conversion byte for byte in every layout.
    #[test]
    fn repeated_pattern_block_matches_the_whole_conversion() {
        let mut compared = Vec::new();
        for representation in ["q8g32s", "q4k", "q5k", "q6k", "iq4g32"] {
            let (source, packet) = pattern_source(representation).unwrap();
            for layout in [Layout::Packet, Layout::Rows16, Layout::Rows8, Layout::Mma16] {
                let Some(element) = Element::stored(representation, layout) else {
                    continue;
                };
                let columns = 512;
                let Some(whole) =
                    pattern_bytes(element, source, &packet, &[2 * PATTERN_BLOCK_ROWS, columns])
                else {
                    continue;
                };
                let block = pattern_bytes(element, source, &packet, &[PATTERN_BLOCK_ROWS, columns])
                    .unwrap();
                assert_eq!(block.repeat(2), whole, "{representation} {layout:?}");
                compared.push((representation, layout));
            }
        }
        // Every production row layout of every fixture format was compared.
        for representation in ["q8g32s", "q4k", "q5k", "q6k", "iq4g32"] {
            assert!(
                compared.contains(&(representation, Layout::Rows16)),
                "{representation} rows16 was not compared: {compared:?}"
            );
        }
    }
}
