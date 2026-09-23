use super::super::*;

pub(super) fn qualification(
    entry: &'static str,
    bindings: &'static str,
    outcome: impl fmt::Display,
) -> CatalogError {
    CatalogError::Qualification {
        path: ExecutionPath::NativeMetal,
        entry,
        bindings: bindings.into(),
        outcome: outcome.to_string(),
    }
}

pub(super) fn qualification_dynamic(
    entry: &'static str,
    bindings: &str,
    outcome: impl fmt::Display,
) -> CatalogError {
    CatalogError::Qualification {
        path: ExecutionPath::NativeMetal,
        entry,
        bindings: bindings.to_owned(),
        outcome: outcome.to_string(),
    }
}

pub(super) fn qualify_attention_stages(
    device: &Device,
    kernels: &AttentionKernels,
    hidden: &Tensor,
    input_norm: &Tensor,
    query_gate_weight: &Tensor,
    key_weight: &Tensor,
    value_weight: &Tensor,
    query_norm: &Tensor,
    key_norm: &Tensor,
    output_weight: &Tensor,
    coordinates: &Tensor,
    rotary_components: &Tensor,
    visible: &Tensor,
    fresh: &Tensor,
    destinations: &Tensor,
    history_key: &mut Tensor,
    history_value: &mut Tensor,
    activation: Element,
    label: &str,
) -> Result<Tensor, CatalogError> {
    let normalized = kernels
        .normalize
        .call(qwen_attention_normalize::Args {
            hidden,
            input_norm,
            epsilon: 1.0e-5,
        })
        .map_err(|e| qualification_dynamic("qwen_attention_normalize", label, e))?
        .value;
    let projected = kernels
        .project
        .call(qwen_attention_project::Args {
            normalized: &normalized,
            query_norm,
            query_gate_weight,
            key_weight,
            value_weight,
        })
        .map_err(|e| qualification_dynamic("qwen_attention_project", label, e))?;
    let prepared = kernels
        .prepare
        .call(qwen_attention_prepare::Args {
            query_gate: &projected.r0,
            key: &projected.r1,
            query_norm,
            key_norm,
            coordinates,
            rotary_components,
            base: 10_000.0,
            epsilon: 1.0e-5,
        })
        .map_err(|e| qualification_dynamic("qwen_attention_prepare", label, e))?;
    let mut accumulator = semantic_zeros(device, Element::f32(), &[1, 1, 4], "attention", label)?;
    let gated = kernels
        .attend
        .call(qwen_attention_attend::Args {
            query: &prepared.r0,
            prepared_key: &prepared.r1,
            value: &projected.r2,
            gate: &prepared.r2,
            visible,
            fresh,
            history_key,
            history_value,
            accumulator: &mut accumulator,
            scale: 0.5,
        })
        .map_err(|e| qualification_dynamic("qwen_attention_attend", label, e))?
        .value;
    kernels
        .output
        .call(qwen_attention_output::Args {
            hidden,
            gated: &gated,
            prepared_key: &prepared.r1,
            value: &projected.r2,
            output_weight,
            destinations,
            history_key,
            history_value,
        })
        .map_err(|e| qualification_dynamic("qwen_attention_output", label, e))
        .map(|out| out.value)
}

pub(super) fn repack_bindings() -> [(Element, Element, &'static str, u64, usize); 5] {
    [
        (
            Element::named("gguf_q8_0").expect("registered external Q8_0 representation"),
            Element::named("q8g32s").expect("registered resident Q8 representation"),
            "E=gguf_q8_0,U=q8g32s",
            32,
            34,
        ),
        (
            Element::named("gguf_q4_k").expect("registered external Q4_K representation"),
            Element::named("q4k").expect("registered resident Q4_K representation"),
            "E=gguf_q4_k,U=q4k",
            256,
            144,
        ),
        (
            Element::named("gguf_q5_k").expect("registered external Q5_K representation"),
            Element::named("q5k").expect("registered resident Q5_K representation"),
            "E=gguf_q5_k,U=q5k",
            256,
            176,
        ),
        (
            Element::named("gguf_q6_k").expect("registered external Q6_K representation"),
            Element::named("q6k").expect("registered resident Q6_K representation"),
            "E=gguf_q6_k,U=q6k",
            256,
            210,
        ),
        (
            Element::named("gguf_iq4_xs").expect("registered external IQ4_XS representation"),
            Element::named("iq4g32").expect("registered resident IQ4_XS representation"),
            "E=gguf_iq4_xs,U=iq4g32",
            256,
            136,
        ),
    ]
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
) -> Result<Tensor, CatalogError> {
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
) -> Result<Tensor, CatalogError> {
    Tensor::zeros(device, element, extents)
        .map_err(|error| qualification_dynamic(entry, bindings, error))
}

pub(super) fn semantic_f32(
    device: &Device,
    extents: &[u64],
    values: &[f32],
    entry: &'static str,
    bindings: &str,
) -> Result<Tensor, CatalogError> {
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
) -> Result<Tensor, CatalogError> {
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
) -> Result<Tensor, CatalogError> {
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
) -> Result<Tensor, CatalogError> {
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
) -> Result<Tensor, CatalogError> {
    if element.dtype().is_some() {
        return semantic_ones(device, element, extents, entry, bindings);
    }
    let mut tensor = semantic_zeros(device, element, extents, entry, bindings)?;
    let bytes = usize::try_from(tensor.storage_bytes())
        .map_err(|_| qualification_dynamic(entry, bindings, "fixture storage exceeds usize"))?;
    let mut pattern = vec![0_u8; bytes];
    match element.name() {
        "q8g32s" => pattern.chunks_exact_mut(34).for_each(|packet| {
            packet[..32].fill(1);
            packet[32..34].copy_from_slice(&0x3c00_u16.to_le_bytes());
        }),
        "q4k" => pattern.chunks_exact_mut(144).for_each(|packet| {
            packet[..128].fill(0x11);
            packet[128..140].fill(1);
            packet[140..142].copy_from_slice(&0x3c00_u16.to_le_bytes());
        }),
        "q5k" => pattern.chunks_exact_mut(176).for_each(|packet| {
            packet[..160].fill(1);
            packet[160..172].fill(1);
            packet[172..174].copy_from_slice(&0x3c00_u16.to_le_bytes());
        }),
        "q6k" => pattern.chunks_exact_mut(210).for_each(|packet| {
            packet[..192].fill(1);
            packet[192..208].fill(1);
            packet[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
        }),
        "iq4g32" => pattern.chunks_exact_mut(160).for_each(|packet| {
            packet[..128].fill(0x11);
            for group in 0..8 {
                packet[128 + group * 4..132 + group * 4].copy_from_slice(&1.0_f32.to_le_bytes());
            }
        }),
        name => {
            return Err(qualification_dynamic(
                entry,
                bindings,
                format!("no non-degenerate fixture exists for {name}"),
            ));
        }
    }
    tensor
        .write_from_host(&pattern)
        .map_err(|error| qualification_dynamic(entry, bindings, error))?;
    Ok(tensor)
}

pub(super) fn require_zero_result(
    tensor: &Tensor,
    entry: &'static str,
    bindings: &str,
) -> Result<(), CatalogError> {
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
) -> Result<(), CatalogError> {
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
) -> Result<(), CatalogError> {
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
) -> Result<(), CatalogError> {
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

pub(super) fn require_dense_values(
    tensor: &Tensor,
    expected: &[f32],
    entry: &'static str,
    bindings: &str,
) -> Result<(), CatalogError> {
    let actual = read_dense_values(tensor, entry, bindings)?;
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(actual, expected)| (actual - expected).abs() > 0.05)
    {
        return Err(qualification_dynamic(
            entry,
            bindings,
            "semantic fixture result mismatch",
        ));
    }
    Ok(())
}

pub(super) fn require_not_dense_values(
    tensor: &Tensor,
    rejected: &[f32],
    entry: &'static str,
    bindings: &str,
) -> Result<(), CatalogError> {
    let actual = read_dense_values(tensor, entry, bindings)?;
    if actual.len() == rejected.len()
        && actual
            .iter()
            .zip(rejected)
            .all(|(actual, rejected)| (actual - rejected).abs() <= 0.01)
    {
        return Err(qualification_dynamic(
            entry,
            bindings,
            "semantic fixture did not exercise the projection",
        ));
    }
    Ok(())
}

pub(super) fn read_dense_values(
    tensor: &Tensor,
    entry: &'static str,
    bindings: &str,
) -> Result<Vec<f32>, CatalogError> {
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
