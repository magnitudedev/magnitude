//! Attention block graph: the normed query | gate | key | value projection
//! (`attention_project`), the fused attention entry of the history codec
//! (head norms, rotary, K/V append at the rows' destinations, attention over
//! the visible history spans and fresh rows, output gate) and the output
//! projection plus residual (or its post-norm tail). Decode row classes use the codec's decode entry
//! (`attention_decode`, `attention_decode_k8v4`); larger classes its prefill
//! entry. Both share one contract, so their ports have the same geometry.
//!
//! Every optional part of the operator is an axis of extent 0 or 1
//! ([`AttentionShape`]): an absent projection segment binds a zero-row view
//! of the query weight, an absent head norm a zero-row view of the block's
//! unit norm row. The rotary table (the coordinate axis, frequency and
//! amplitude of every rotated pair) and the unit row are graph constants,
//! bound once with the weights.

use crate::operators;
use crate::operators::output::{post_norm, CheckedPostNormEntries, PostNormShape, TailEntries};
use crate::programs::graph::{draft::GraphDraft, GraphError};
use crate::programs::native_target_graph::ScaledWeight;
use crate::{
    native::{AttentionHistoryKernels, AttentionKernels},
    programs::native_constants::GraphConstant,
};
use crate::{AttentionBinding, AttentionShape, SublayerTail};
use magnitude_family_contracts::{Attention, Rotary, WeightKind};
use magnitude_kernels::{
    attention_decode, attention_decode_k8v4, attention_output, attention_prefill,
    attention_prefill_k8v4, attention_project,
};
use magnitude_state::KvCodec;
use magnitude_state::AFFINE_GROUP;
use seismic::{
    Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor, WorkflowTensorRef,
    WorkflowTensorView,
};

pub(crate) struct AttentionGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, attention_project::Entry>,
    pub history: AttentionHistoryEntries<'a, G>,
    pub output: TailEntries<'a, G, attention_output::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for AttentionGraphEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for AttentionGraphEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

pub(crate) enum AttentionHistoryEntries<'a, G: GraphDraft + 'a> {
    Dense {
        decode: G::Binding<'a, attention_decode::Entry>,
        verify: Option<G::Binding<'a, attention_decode::Entry>>,
        prefill: G::Binding<'a, attention_prefill::Entry>,
    },
    AffineK8V4 {
        decode: G::Binding<'a, attention_decode_k8v4::Entry>,
        verify: Option<G::Binding<'a, attention_decode_k8v4::Entry>>,
        verify_four: Option<G::Binding<'a, attention_decode_k8v4::Entry>>,
        verify_eight: Option<G::Binding<'a, attention_decode_k8v4::Entry>>,
        prefill: G::Binding<'a, attention_prefill_k8v4::Entry>,
        /// The prefill of a class that lists its launch's history row tiles
        /// (`StateResourcePlan::lists_history_tiles`).
        prefill_listed: Option<G::Binding<'a, attention_prefill_k8v4::Entry>>,
    },
}

impl<'a, G: GraphDraft + 'a> Copy for AttentionHistoryEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for AttentionHistoryEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a> From<&'a AttentionKernels> for AttentionGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a AttentionKernels) -> Self {
        let history = match &kernels.history {
            AttentionHistoryKernels::Dense {
                decode,
                verify,
                prefill,
            } => AttentionHistoryEntries::Dense {
                decode,
                verify: verify.as_ref(),
                prefill,
            },
            AttentionHistoryKernels::AffineK8V4 {
                decode,
                verify,
                verify_four,
                verify_eight,
                prefill,
                prefill_listed,
            } => AttentionHistoryEntries::AffineK8V4 {
                decode,
                verify: verify.as_ref(),
                verify_four: verify_four.as_ref(),
                verify_eight: verify_eight.as_ref(),
                prefill,
                prefill_listed: prefill_listed.as_ref(),
            },
        };
        Self {
            project: &kernels.project,
            history,
            output: (&kernels.output).into(),
        }
    }
}

/// Entry element assignments from the same program binding used by native
/// preparation. The checked graph obtains all shapes from the entries.
pub(crate) struct CheckedAttentionEntries {
    project: [(&'static str, Element); 6],
    mix: [(&'static str, Element); 1],
    output: [(&'static str, Element); 2],
    post_norm: Option<CheckedPostNormEntries>,
    history: KvCodec,
    /// Whether the graphs may list their launch's history row tiles
    /// (`StateResourcePlan::lists_history_tiles`, for a program whose
    /// classes have the listing axis).
    lists: bool,
}

impl CheckedAttentionEntries {
    pub(crate) fn new(binding: AttentionBinding, lists: bool) -> Self {
        Self {
            lists: lists && binding.history == KvCodec::AffineK8V4,
            project: [
                ("NW", binding.norm),
                ("QW", binding.query),
                ("GW", binding.gate),
                ("KW", binding.key),
                ("VW", binding.value),
                ("A", binding.activation),
            ],
            mix: [("A", binding.activation)],
            output: [("OW", binding.output), ("A", binding.activation)],
            post_norm: match binding.tail {
                SublayerTail::Residual => None,
                SublayerTail::PostNorm { norm, .. } => Some(CheckedPostNormEntries::new(
                    binding.output,
                    binding.activation,
                    norm,
                )),
            },
            history: binding.history,
        }
    }

    pub(crate) fn entries(&self) -> Result<AttentionGraphEntries<'_, NativeGraphMetadata>, String> {
        let history = match self.history {
            KvCodec::Dense => AttentionHistoryEntries::Dense {
                decode: &self.mix[..],
                verify: None,
                prefill: &self.mix[..],
            },
            KvCodec::AffineK8V4 => AttentionHistoryEntries::AffineK8V4 {
                decode: &self.mix[..],
                verify: None,
                verify_four: None,
                verify_eight: None,
                prefill: &self.mix[..],
                prefill_listed: self.lists.then_some(&self.mix[..]),
            },
            KvCodec::RotatedK4V4 => {
                return Err("rotated K4/V4 has no native attention entry".into())
            }
        };
        Ok(AttentionGraphEntries {
            project: &self.project,
            history,
            output: match &self.post_norm {
                None => TailEntries::Residual(&self.output[..]),
                Some(post_norm) => TailEntries::PostNorm(post_norm.entries()),
            },
        })
    }
}

/// Row classes up to this size attend with the decode entry; larger classes
/// use the prefill entry.
pub(crate) const DECODE_ROWS: u64 = 8;

/// Whether a `rows`-row class attends with the decode entry.
pub(crate) fn decodes(rows: u64) -> bool {
    rows <= DECODE_ROWS
}

/// F16 elements of one head vector's affine (scale, zero) pairs: one pair per
/// codec group of a `width`-wide head.
pub(crate) const fn affine_coefficients(width: u64) -> u64 {
    2 * width / AFFINE_GROUP as u64
}

/// The block's weight tensors, as ports of the graph being built. The
/// operator's form decides which are present.
pub(crate) struct AttentionWeights {
    pub input_norm: WorkflowTensor,
    /// The query projection (with interleaved gate rows when gated so).
    pub query: WorkflowTensor,
    pub gate: Option<WorkflowTensor>,
    pub key: Option<WorkflowTensor>,
    pub value: Option<WorkflowTensor>,
    pub query_norm: Option<WorkflowTensor>,
    pub key_norm: Option<WorkflowTensor>,
    pub output: Option<WorkflowTensor>,
    /// The sublayer's post-norm weight, when its tail is a post-norm.
    pub post_norm: Option<WorkflowTensor>,
}

/// One attention block at one graph class.
pub(crate) struct AttentionBlock<'a> {
    pub rows: u64,
    pub segments: u64,
    pub history_rows: u64,
    pub slab_rows: u32,
    /// The history row tiles one launch of the class can see
    /// (`HistoryStorePlan::launch_tiles` of its request slots).
    pub history_tiles: u64,
    pub shape: AttentionShape,
    pub operator: &'a Attention,
    /// The input norm's epsilon.
    pub epsilon: f32,
    /// The head norms' one epsilon.
    pub head_epsilon: f32,
    /// The post-norm's epsilon, read only by a post-norm tail.
    pub post_norm_epsilon: f32,
    /// The post-norm's output scale: 1, or the layer's output scale.
    pub post_norm_scale: f32,
    pub activation: Element,
    /// Draft priming only publishes K/V history; its attention output is unused.
    pub inject_only: bool,
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
    /// The history row tiles the launch's rows see, for an entry that takes
    /// them (the affine prefill).
    pub history_tiles: Option<HistoryTilesPort>,
}

/// The `history_tiles` input of a graph class that lists the history row
/// tiles its launch's rows see, and the tiles it lists.
#[derive(Clone)]
pub(crate) struct HistoryTilesPort {
    pub port: NativePort,
    pub tiles: usize,
}

/// The history row tiles the rows of a launch see, from its encoded
/// `visible` table (little-endian `[start, end)` pairs): distinct and
/// ascending, then -1 up to `tiles` entries, as little-endian bytes. `None`
/// when they see more than `tiles`: the launch takes the class that lists
/// none.
pub(crate) fn history_tile_bytes(visible: &[u8], tiles: usize) -> Option<Vec<u8>> {
    let spans = visible.chunks_exact(8).map(|span| {
        [
            i32::from_le_bytes([span[0], span[1], span[2], span[3]]),
            i32::from_le_bytes([span[4], span[5], span[6], span[7]]),
        ]
    });
    magnitude_batching::history_tiles(spans, tiles)
        .map(|tiles| tiles.iter().flat_map(|tile| tile.to_le_bytes()).collect())
}


/// The block's attention weights in the order and presence its operator's
/// form and its sublayer tail define, from a lookup of a role's port.
pub(crate) fn attention_weights(
    shape: &AttentionShape,
    operator: &Attention,
    needs_output: bool,
    post_norm: bool,
    mut weight: impl FnMut(WeightKind) -> Result<WorkflowTensor, GraphError>,
) -> Result<AttentionWeights, GraphError> {
    let mut present = |present: bool, kind| present.then(|| weight(kind)).transpose();
    Ok(AttentionWeights {
        input_norm: present(true, WeightKind::InputNorm)?.ok_or("input norm")?,
        query: present(true, operators::attention::query_kind(operator))?.ok_or("query")?,
        gate: present(shape.gate_rows() > 0, WeightKind::AttentionGate)?,
        key: present(shape.key_rows() > 0, WeightKind::Key)?,
        value: present(shape.value_rows() > 0, WeightKind::Value)?,
        query_norm: present(shape.head_norm > 0, WeightKind::QueryNorm)?,
        // A Shared layer projects no keys, so it has no key norm.
        key_norm: present(shape.head_norm > 0 && shape.fresh > 0, WeightKind::KeyNorm)?,
        output: present(needs_output, WeightKind::AttentionOutput)?,
        post_norm: present(post_norm, WeightKind::PostNorm)?,
    })
}

pub(crate) fn attention<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    kernels: AttentionGraphEntries<'a, G>,
    weights: &AttentionWeights,
    constants: &mut Vec<GraphConstant>,
    hidden: &WorkflowTensor,
    block: AttentionBlock<'_>,
) -> Result<(WorkflowTensor, AttentionStatePorts, AttentionControlPorts), GraphError> {
    let shape = block.shape;
    let (rows, width, heads) = (block.rows, shape.width, shape.heads());
    // An absent projection segment reads zero rows of the query weight.
    let empty_weight = weights.query.slice_leading(0, 0);
    let projected = graph.enqueue(
        kernels.project,
        &shape.project_dimensions(rows),
        attention_project::WorkflowArgs {
            hidden: hidden.into(),
            input_norm: (&weights.input_norm).into(),
            query_weight: (&weights.query).into(),
            gate_weight: segment(&weights.gate, &empty_weight),
            key_weight: segment(&weights.key, &empty_weight),
            value_weight: segment(&weights.value, &empty_weight),
            epsilon: block.epsilon,
            project_mode: if block.inject_only { 1 } else { 0 },
        },
    )?;
    // The mix entries read the projection per head.
    let projected = ProjectedRows {
        query: projected
            .r0
            .reshape(&[rows, heads, width + shape.interleaved_gate]),
        gate: projected.r1.reshape(&[rows, heads, shape.separate_gate]),
        key: projected
            .r2
            .reshape(&[shape.fresh, rows, shape.kv_heads * width]),
        value: if shape.projected_value {
            projected
                .r3
                .reshape(&[shape.fresh, rows, shape.kv_heads * width])
        } else {
            projected
                .r2
                .reshape(&[shape.fresh, rows, shape.kv_heads * width])
        },
    };
    let (attended, state, controls) = mix(
        graph,
        kernels.history,
        (&weights.query_norm, &weights.key_norm),
        constants,
        &projected,
        &block,
    )?;
    if block.inject_only {
        return Ok((hidden.clone(), state, controls));
    }
    let output_weight = weights
        .output
        .as_ref()
        .ok_or("attention output weight is absent")?;
    let mixed = match (kernels.output, &weights.post_norm) {
        (TailEntries::Residual(output), None) => {
            graph
                .enqueue(
                    output,
                    &[("M", rows), ("D", shape.hidden), ("Q", heads), ("W", width)],
                    attention_output::WorkflowArgs {
                        hidden: hidden.into(),
                        gated: (&attended).into(),
                        output_weight: output_weight.into(),
                    },
                )?
                .value
        }
        (TailEntries::PostNorm(entries), Some(norm)) => {
            let out_rows = GraphConstant::identity_for_class(graph, rows, Some("M"))?;
            let absent_scale = GraphConstant::absent_scale(graph, constants)?;
            let mixed = post_norm(
                graph,
                entries,
                hidden,
                (&attended.reshape(&[rows, heads * width])).into(),
                &ScaledWeight::unscaled(output_weight.clone(), absent_scale),
                norm,
                out_rows.port().tensor(),
                PostNormShape {
                    rows,
                    out: rows,
                    inputs: heads * width,
                    outputs: shape.hidden,
                },
                block.post_norm_epsilon,
                block.post_norm_scale,
            )?;
            constants.push(out_rows);
            mixed
        }
        _ => return Err("attention output entries disagree with its post-norm weight".into()),
    };
    Ok((mixed, state, controls))
}

/// The per-head rows the history codec's attention entry reads: the query
/// (with its interleaved gate), the separate gate, and the fresh keys and
/// values.
#[derive(Clone)]
pub(crate) struct ProjectedRows {
    /// `[rows, heads, width + interleaved gate]`.
    pub query: WorkflowTensorView,
    /// `[rows, heads, separate gate]`.
    pub gate: WorkflowTensorView,
    /// `[fresh, rows, kv heads · width]`.
    pub key: WorkflowTensorView,
    pub value: WorkflowTensorView,
}

/// The history codec's fused attention entry over projected rows: head
/// norms, rotary, K/V append at the rows' destinations and attention over
/// the visible spans and fresh rows. Its rotary table and unit norm row join
/// `constants`.
pub(crate) fn mix<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    history: AttentionHistoryEntries<'a, G>,
    (query_norm, key_norm): (&Option<WorkflowTensor>, &Option<WorkflowTensor>),
    constants: &mut Vec<GraphConstant>,
    projected: &ProjectedRows,
    block: &AttentionBlock<'_>,
) -> Result<(WorkflowTensor, AttentionStatePorts, AttentionControlPorts), GraphError> {
    let shape = block.shape;
    let (rows, width) = (block.rows, shape.width);
    let dimensions = shape.mix_dimensions(rows, block.history_rows, block.segments);
    let input = |graph: &mut G, name: &str| match &history {
        AttentionHistoryEntries::Dense { decode, .. } => {
            graph.input_for(*decode, name, &dimensions)
        }
        AttentionHistoryEntries::AffineK8V4 { decode, .. } => {
            graph.input_for(*decode, name, &dimensions)
        }
    };
    let decode = decodes(rows);
    // The affine prefill of a class that lists its launch's history row
    // tiles sizes its per-call history storage by them, never by the store's
    // reservation; a class that lists none (and an injection-only route,
    // which attends to nothing) reads the history in place.
    let listed = match &history {
        AttentionHistoryEntries::AffineK8V4 {
            prefill_listed: Some(kernel),
            ..
        } if !decode && !block.inject_only && block.history_tiles > 0 => Some(*kernel),
        _ => None,
    };
    let tile_dimensions = dimensions
        .iter()
        .copied()
        .chain([
            ("L", u64::from(listed.is_some())),
            ("HT", listed.map_or(1, |_| block.history_tiles)),
        ])
        .collect::<Vec<_>>();
    let history_tiles = listed
        .map(|kernel| {
            Ok::<_, GraphError>(HistoryTilesPort {
                port: graph.input_for(kernel, "history_tiles", &tile_dimensions)?,
                tiles: usize::try_from(block.history_tiles)
                    .map_err(|_| "history tile count exceeds host domain")?,
            })
        })
        .transpose()?;
    let controls = AttentionControlPorts {
        coordinates: input(graph, "coordinates")?,
        visible: input(graph, "visible")?,
        fresh: input(graph, "fresh")?,
        destinations: input(graph, "destinations")?,
        history_tiles,
    };
    let rotary = &block.operator.rotary;
    let components = GraphConstant::i32(graph, &rotary_components(rotary)?)?;
    let frequencies = GraphConstant::f32(graph, &rotary_frequencies(rotary))?;
    let amplitudes = GraphConstant::f32(graph, &rotary_amplitudes(rotary))?;
    // The unit norm row: the unweighted value norm, and (zero rows of it)
    // every absent head norm.
    let unit = GraphConstant::f32_shaped(graph, &[1, width], &vec![1.0; width as usize])?;
    let absent_norm = unit.port().tensor().slice_leading(0, 0);
    let head_norm = |weight: &Option<WorkflowTensor>| match weight {
        Some(weight) => weight.reshape(&[1, width]),
        None => absent_norm.clone(),
    };
    let query_norm = head_norm(query_norm);
    // The entries take equal query and key norm rows; a Shared layer's
    // unread key norm port takes its query norm.
    let key_norm = if shape.fresh == 0 {
        query_norm.clone()
    } else {
        head_norm(key_norm)
    };
    let value_norm = unit.port().tensor().slice_leading(0, shape.value_norm);
    let ProjectedRows {
        query,
        gate,
        key,
        value,
    } = projected.clone();
    let mut plane = |element: Element, elements: u64| {
        graph.port(element, &[block.history_rows, shape.kv_heads, elements])
    };
    let mut planes = match &history {
        AttentionHistoryEntries::Dense { .. } => vec![
            plane(block.activation, width)?,
            plane(block.activation, width)?,
        ],
        AttentionHistoryEntries::AffineK8V4 { .. } => vec![
            plane(Element::u32(), width / 4)?,
            plane(Element::f16(), affine_coefficients(width))?,
            plane(Element::u32(), width / 8)?,
            plane(Element::f16(), affine_coefficients(width))?,
        ],
    };
    let scale = block.operator.scale as f32;
    // -1 marks the injection-only route. Metal still runs the prepare/append
    // launch, while its attention and merge launches have no work to do.
    let gate_function = if block.inject_only {
        -1
    } else {
        operators::attention::gate_function(block.operator)
    };
    // Every entry takes the same arguments but its history planes.
    macro_rules! mix {
        ($kernel:expr, $module:ident, $($plane:ident),*) => {
            mix!(@enqueue $kernel, $module, &dimensions, {}, $($plane),*)
        };
        (@enqueue $kernel:expr, $module:ident, $dimensions:expr, {$($extra:ident: $value:expr),*},
            $($plane:ident),*) => {{
            let [$($plane),*] = planes.as_mut_slice() else {
                return Err("attention history planes disagree with the entry".into());
            };
            graph
                .enqueue(
                    *$kernel,
                    $dimensions,
                    $module::WorkflowArgs {
                        query: (&query).into(),
                        gate: (&gate).into(),
                        key: (&key).into(),
                        value: (&value).into(),
                        query_norm: (&query_norm).into(),
                        key_norm: (&key_norm).into(),
                        value_norm: (&value_norm).into(),
                        rotary_components: components.port().tensor().into(),
                        rotary_frequencies: frequencies.port().tensor().into(),
                        rotary_amplitudes: amplitudes.port().tensor().into(),
                        coordinates: controls.coordinates.tensor().into(),
                        visible: controls.visible.tensor().into(),
                        fresh: controls.fresh.tensor().into(),
                        destinations: controls.destinations.tensor().into(),
                        $($extra: $value,)*
                        $($plane: $plane.tensor_mut().into(),)*
                        epsilon: block.head_epsilon,
                        scale,
                        gate_function,
                        slab_rows: block.slab_rows,
                    },
                )?
                .value
        }};
    }
    let attended = match &history {
        AttentionHistoryEntries::Dense {
            verify: Some(kernel),
            ..
        } if decode && rows > 1 => {
            mix!(kernel, attention_decode, history_key, history_value)
        }
        AttentionHistoryEntries::Dense { decode: kernel, .. } if decode => {
            mix!(kernel, attention_decode, history_key, history_value)
        }
        AttentionHistoryEntries::Dense {
            prefill: kernel, ..
        } => {
            mix!(kernel, attention_prefill, history_key, history_value)
        }
        AttentionHistoryEntries::AffineK8V4 {
            verify_four: Some(kernel),
            ..
        } if decode && rows == 4 => mix!(
            kernel,
            attention_decode_k8v4,
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
        ),
        AttentionHistoryEntries::AffineK8V4 {
            verify_eight: Some(kernel),
            ..
        } if decode && rows > 4 => mix!(
            kernel,
            attention_decode_k8v4,
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
        ),
        AttentionHistoryEntries::AffineK8V4 {
            verify: Some(kernel),
            ..
        } if decode && rows > 1 => mix!(
            kernel,
            attention_decode_k8v4,
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
        ),
        AttentionHistoryEntries::AffineK8V4 { decode: kernel, .. } if decode => mix!(
            kernel,
            attention_decode_k8v4,
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
        ),
        AttentionHistoryEntries::AffineK8V4 { prefill, .. } => {
            // No list: zero rows of an I32 table.
            let unlisted = controls
                .destinations
                .tensor()
                .slice_leading(0, 0)
                .reshape(&[0, 1]);
            let kernel = listed.as_ref().unwrap_or(prefill);
            mix!(
            @enqueue kernel,
            attention_prefill_k8v4,
            &tile_dimensions,
            {
                history_tiles: match &controls.history_tiles {
                    Some(tiles) => tiles.port.tensor().into(),
                    None => (&unlisted).into(),
                }
            },
            history_key_codes,
            history_key_coefficients,
            history_value_codes,
            history_value_coefficients
            )
        }
    };
    constants.extend([components, frequencies, amplitudes, unit]);
    Ok((attended, AttentionStatePorts { planes }, controls))
}

/// A projection segment's weight, or zero rows of the query weight when the
/// form has no such segment.
fn segment<'t>(
    weight: &'t Option<WorkflowTensor>,
    empty: &'t WorkflowTensorView,
) -> WorkflowTensorRef<'t> {
    match weight {
        Some(weight) => weight.into(),
        None => empty.into(),
    }
}

/// Per rotary pair, the coordinate axis that drives it: the interleaved
/// multi-axis layout assigns pairs to axes round-robin until each axis's
/// section is exhausted; a table rotates by the text position, axis 0.
pub(crate) fn rotary_components(rotary: &Rotary) -> Result<Vec<i32>, String> {
    let (width, sections, axis_pattern) = match rotary {
        Rotary::None => return Ok(Vec::new()),
        Rotary::Table { pairs, .. } => return Ok(vec![0; pairs.len()]),
        Rotary::Interleaved {
            width,
            sections,
            axis_pattern,
            ..
        } => (width, sections, axis_pattern),
    };
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

/// Per rotary pair `p` of `P`, its angular frequency: `base^(-2p / 2P)` for
/// the interleaved layout, evaluated in f64 and rounded once to f32, or the
/// table's own frequency.
pub(crate) fn rotary_frequencies(rotary: &Rotary) -> Vec<f32> {
    match rotary {
        Rotary::None => Vec::new(),
        Rotary::Interleaved { width, base, .. } => {
            let pairs = width / 2;
            (0..pairs)
                .map(|pair| base.powf(-((2 * pair) as f64) / (2 * pairs) as f64) as f32)
                .collect()
        }
        Rotary::Table { pairs, .. } => pairs.iter().map(|pair| pair.frequency as f32).collect(),
    }
}

/// Per rotary pair, the amplitude its rotation is scaled by (1 is exact
/// identity scaling).
pub(crate) fn rotary_amplitudes(rotary: &Rotary) -> Vec<f32> {
    match rotary {
        Rotary::None => Vec::new(),
        Rotary::Interleaved { width, .. } => vec![1.0; (width / 2) as usize],
        Rotary::Table { pairs, .. } => pairs.iter().map(|pair| pair.amplitude as f32).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A launch takes the class that lists its history row tiles exactly
    /// when its rows see at most that class's tiles: one request with a long
    /// history does, however many rows and spans it has; several requests
    /// whose histories together exceed it take the class that lists none.
    #[test]
    fn a_launch_lists_its_history_tiles_when_they_fit_one_request() {
        let visible = |spans: &[[i32; 2]]| {
            spans
                .iter()
                .flatten()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let listed = |bytes: Vec<u8>| {
            bytes
                .chunks_exact(4)
                .map(|tile| i32::from_le_bytes([tile[0], tile[1], tile[2], tile[3]]))
                .collect::<Vec<_>>()
        };
        // One request of 64 rows over a 16,384-row history in two spans, the
        // second at a lower address: 64 tiles of 256 rows.
        let request = [[40_960, 49_152], [8_192, 16_384]];
        let rows = std::iter::repeat(request).take(64).flatten().collect::<Vec<_>>();
        let tiles = listed(history_tile_bytes(&visible(&rows), 66).unwrap());
        assert_eq!(tiles[..64], (32..64).chain(160..192).collect::<Vec<_>>());
        assert_eq!(tiles[64..], [-1, -1]);
        // A second request with as long a history of its own no longer fits.
        let both = rows
            .iter()
            .copied()
            .chain([[65_536, 73_728], [90_112, 98_304]])
            .collect::<Vec<_>>();
        assert_eq!(history_tile_bytes(&visible(&both), 66), None);
        // A fork sharing the first request's rows adds no tile.
        let fork = rows
            .iter()
            .copied()
            .chain([[40_960, 45_056], [0, 0]])
            .collect::<Vec<_>>();
        assert_eq!(listed(history_tile_bytes(&visible(&fork), 66).unwrap()), tiles);
        // Rows that see no history list nothing.
        assert_eq!(listed(history_tile_bytes(&visible(&[[0, 0]]), 2).unwrap()), [-1, -1]);
    }

    #[test]
    fn rotary_components_interleave_axes_with_section_cutoffs() {
        let rotary = Rotary::Interleaved {
            width: 14,
            base: 10_000.0,
            sections: vec![4, 2, 1, 0],
            axis_pattern: vec![0, 1, 2],
        };
        assert_eq!(rotary_components(&rotary).unwrap(), [0, 1, 2, 0, 1, 0, 0]);
    }

    #[test]
    fn rotary_frequencies_fall_geometrically_from_one() {
        let rotary = Rotary::Interleaved {
            width: 4,
            base: 10_000.0,
            sections: vec![2, 0, 0, 0],
            axis_pattern: vec![0],
        };
        assert_eq!(rotary_frequencies(&rotary), [1.0, 0.01]);
        assert_eq!(rotary_amplitudes(&rotary), [1.0, 1.0]);
    }
}
