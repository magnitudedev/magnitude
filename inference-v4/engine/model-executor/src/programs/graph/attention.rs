//! Gated attention block graph: the normed Q/K/V projection, the fused
//! attention entry (q/k norms, partial M-RoPE, K/V append at the rows'
//! destinations, attention over the visible history spans and fresh rows,
//! sigmoid gate) and the output projection plus residual. Decode row classes
//! use the history codec's decode entry (`gated_attention_decode`,
//! `gated_attention_decode_k8v4`); larger classes its prefill entry. Both share
//! one contract, so their ports have the same geometry.
//!
//! The rotary table (the coordinate axis and frequency of every rotated pair)
//! is a graph constant, bound once with the weights.

use crate::{
    native::{AttentionHistoryKernels, AttentionKernels},
    programs::native_constants::GraphConstant,
};
use magnitude_model_contracts::RotarySemantics;
use magnitude_model_kernels::{
    gated_attention_decode, gated_attention_decode_k8v4, attention_output,
    gated_attention_prefill, gated_attention_prefill_k8v4, gated_attention_project,
};
use magnitude_model_state::AFFINE_GROUP;
use seismic::{Element, NativeGraph, NativePort, WorkflowTensor};

/// Row classes up to this size attend with the decode entry; larger classes
/// use the prefill entry.
pub(crate) const DECODE_ROWS: u64 = 8;

/// F16 elements of one head vector's affine (scale, zero) pairs: one pair per
/// codec group of a `width`-wide head.
pub(crate) const fn affine_coefficients(width: u64) -> u64 {
    2 * width / AFFINE_GROUP as u64
}

/// The block's weight tensors, as ports of the graph being built.
pub(crate) struct AttentionWeights {
    pub input_norm: WorkflowTensor,
    pub query_norm: WorkflowTensor,
    pub key_norm: WorkflowTensor,
    pub query_gate: WorkflowTensor,
    pub key: WorkflowTensor,
    pub value: WorkflowTensor,
    pub output: WorkflowTensor,
}

/// Geometry of one attention block at one graph class.
pub(crate) struct AttentionBlock<'a> {
    pub rows: u64,
    pub segments: u64,
    pub history_rows: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub width: u64,
    pub rotary: &'a RotarySemantics,
    pub epsilon: f32,
    pub activation: Element,
}

/// The layer's history planes, bound to the state store's planes per run, in
/// the order of the history codec's plane descriptors (the key's planes,
/// then the value's): dense key and value, or affine key codes, key
/// coefficients, value codes and value coefficients.
#[derive(Clone)]
pub(crate) struct AttentionStatePorts {
    pub planes: Vec<NativePort>,
}

/// Per-run row tables: rotary coordinates, visible history spans, fresh
/// spans over the batch rows and the history row each row's K/V appends to.
#[derive(Clone)]
pub(crate) struct AttentionControlPorts {
    pub coordinates: NativePort,
    pub visible: NativePort,
    pub fresh: NativePort,
    pub destinations: NativePort,
}

pub(crate) fn attention(
    graph: &mut NativeGraph,
    kernels: &AttentionKernels,
    weights: &AttentionWeights,
    constants: &mut Vec<GraphConstant>,
    hidden: &WorkflowTensor,
    block: AttentionBlock<'_>,
) -> Result<(WorkflowTensor, AttentionStatePorts, AttentionControlPorts), String> {
    let RotarySemantics::Interleaved {
        width: rotary_width,
        ..
    } = block.rotary;
    let pairs = rotary_width / 2;
    let group = block.heads / block.kv_heads;
    let projected = graph
        .enqueue(
            &kernels.project,
            gated_attention_project::WorkflowArgs {
                hidden: hidden.into(),
                input_norm: (&weights.input_norm).into(),
                query_norm: (&weights.query_norm).into(),
                query_gate_weight: (&weights.query_gate).into(),
                key_weight: (&weights.key).into(),
                value_weight: (&weights.value).into(),
                epsilon: block.epsilon,
            },
        )
        .map_err(|error| error.to_string())?;
    let dimensions = [
        ("M", block.rows),
        ("T", block.history_rows),
        ("KV", block.kv_heads),
        ("G", group),
        ("P", pairs),
        ("S", block.width - 2 * pairs),
        ("R", block.segments),
    ];
    let input = |graph: &mut NativeGraph, name: &str| {
        match &kernels.history {
            AttentionHistoryKernels::Dense { decode, .. } => {
                graph.input_for(decode, name, &dimensions)
            }
            AttentionHistoryKernels::AffineK8V4 { decode, .. } => {
                graph.input_for(decode, name, &dimensions)
            }
        }
        .map_err(|error| error.to_string())
    };
    let controls = AttentionControlPorts {
        coordinates: input(graph, "coordinates")?,
        visible: input(graph, "visible")?,
        fresh: input(graph, "fresh")?,
        destinations: input(graph, "destinations")?,
    };
    let components = GraphConstant::i32(graph, &rotary_components(block.rotary)?)?;
    let frequencies = GraphConstant::f32(graph, &rotary_frequencies(block.rotary))?;
    let mut plane = |element: Element, elements: u64| {
        graph
            .port(element, &[block.history_rows, block.kv_heads, elements])
            .map_err(|error| error.to_string())
    };
    let mut planes = match &kernels.history {
        AttentionHistoryKernels::Dense { .. } => vec![
            plane(block.activation, block.width)?,
            plane(block.activation, block.width)?,
        ],
        AttentionHistoryKernels::AffineK8V4 { .. } => vec![
            plane(Element::u32(), block.width / 4)?,
            plane(Element::f16(), affine_coefficients(block.width))?,
            plane(Element::u32(), block.width / 8)?,
            plane(Element::f16(), affine_coefficients(block.width))?,
        ],
    };
    let scale = 1.0 / (block.width as f32).sqrt();
    // Every entry takes the same arguments but its history planes.
    macro_rules! mix {
        ($kernel:expr, $module:ident, $($plane:ident),*) => {{
            let [$($plane),*] = planes.as_mut_slice() else {
                return Err("attention history planes disagree with the entry".into());
            };
            graph
                .enqueue(
                    $kernel,
                    $module::WorkflowArgs {
                        query_gate: (&projected.r0).into(),
                        key: (&projected.r1).into(),
                        value: (&projected.r2).into(),
                        query_norm: (&weights.query_norm).into(),
                        key_norm: (&weights.key_norm).into(),
                        rotary_components: components.port().tensor().into(),
                        rotary_frequencies: frequencies.port().tensor().into(),
                        coordinates: controls.coordinates.tensor().into(),
                        visible: controls.visible.tensor().into(),
                        fresh: controls.fresh.tensor().into(),
                        destinations: controls.destinations.tensor().into(),
                        $($plane: $plane.tensor_mut().into(),)*
                        epsilon: block.epsilon,
                        scale,
                    },
                )
                .map_err(|error| error.to_string())?
                .value
        }};
    }
    let decode = block.rows <= DECODE_ROWS;
    let gated = match &kernels.history {
        AttentionHistoryKernels::Dense { decode: kernel, .. } if decode => {
            mix!(kernel, gated_attention_decode, history_key, history_value)
        }
        AttentionHistoryKernels::Dense { prefill: kernel, .. } => {
            mix!(kernel, gated_attention_prefill, history_key, history_value)
        }
        AttentionHistoryKernels::AffineK8V4 { decode: kernel, .. } if decode => mix!(
            kernel,
            gated_attention_decode_k8v4,
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
        ),
        AttentionHistoryKernels::AffineK8V4 { prefill: kernel, .. } => mix!(
            kernel,
            gated_attention_prefill_k8v4,
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
        ),
    };
    let mixed = graph
        .enqueue(
            &kernels.output,
            attention_output::WorkflowArgs {
                hidden: hidden.into(),
                gated: (&gated).into(),
                output_weight: (&weights.output).into(),
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    constants.push(components);
    constants.push(frequencies);
    Ok((mixed, AttentionStatePorts { planes }, controls))
}

/// Per rotary pair, the coordinate axis that drives it: the interleaved
/// multi-axis layout assigns pairs to axes round-robin until each axis's
/// section is exhausted.
pub(crate) fn rotary_components(rotary: &RotarySemantics) -> Result<Vec<i32>, String> {
    let RotarySemantics::Interleaved {
        width,
        sections,
        axis_pattern,
        ..
    } = rotary;
    let pairs = usize::try_from(width / 2).map_err(|_| "rotary width exceeds host domain")?;
    let first = *axis_pattern.first().ok_or("rotary axis pattern is empty")?;
    if axis_pattern.len() == 1 {
        return Ok(vec![i32::from(first); pairs]);
    }
    if axis_pattern.len() != 3
        || sections.len() < 3
        || sections[3..].iter().any(|section| *section != 0)
    {
        return Err("unsupported rotary axis and section mapping".into());
    }
    let axis_one_end = sections[1]
        .checked_mul(3)
        .ok_or("rotary section exceeds host domain")?;
    let axis_two_end = sections[2]
        .checked_mul(3)
        .ok_or("rotary section exceeds host domain")?;
    Ok((0..pairs)
        .map(|index| {
            let index = index as u64;
            let axis = if index % 3 == 1 && index < axis_one_end {
                1
            } else if index % 3 == 2 && index < axis_two_end {
                2
            } else {
                0
            };
            i32::from(axis_pattern[axis])
        })
        .collect())
}

/// Per rotary pair `p` of `P`, its angular frequency `base^(-2p / 2P)`,
/// evaluated in f64 and rounded once to f32.
pub(crate) fn rotary_frequencies(rotary: &RotarySemantics) -> Vec<f32> {
    let RotarySemantics::Interleaved { width, base, .. } = rotary;
    let pairs = width / 2;
    (0..pairs)
        .map(|pair| base.powf(-((2 * pair) as f64) / (2 * pairs) as f64) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotary_components_interleave_axes_with_section_cutoffs() {
        let rotary = RotarySemantics::Interleaved {
            width: 14,
            base: 10_000.0,
            sections: vec![4, 2, 1, 0],
            axis_pattern: vec![0, 1, 2],
        };
        assert_eq!(rotary_components(&rotary).unwrap(), [0, 1, 2, 0, 1, 0, 0]);
    }

    #[test]
    fn rotary_frequencies_fall_geometrically_from_one() {
        let rotary = RotarySemantics::Interleaved {
            width: 4,
            base: 10_000.0,
            sections: vec![2, 0, 0, 0],
            axis_pattern: vec![0],
        };
        assert_eq!(rotary_frequencies(&rotary), [1.0, 0.01]);
    }
}
