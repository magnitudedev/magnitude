//! Decode demand derived directly from `SizingDraft`'s recorded decode-regime
//! launch census (`TargetFamilySizingCensus` and `ReadoutSizingCensus`), with
//! zero `.seismicbundle` decoding and zero hand-written per-operator mirrors.
//!
//! Each recorded `SizedCall` carries its entry name, compiled `EntryContract`
//! sizing evaluator (for `Launches::at(rows)`), and tagged weight ports
//! (`SizedBoundPort::is_weight`). This module maps the ordered decode calls of
//! entry, per-layer entry, target blocks, and readout into the `DecodeDemand`
//! consumed by `estimate.rs` and `magnitude_cost_model::Census`.

use super::launches::{Launches, Words};
use super::AssessmentError;
use crate::operators::parallel::DenseBesideRouted;
use crate::operators::routed::streamed;
use crate::operators::streams::{BlockStreams, SinkhornBinding};
use crate::programs::graph::sizing::{
    DraftFamilySizingCensus, ReadoutSizingCensus, SizedCall, TargetFamilySizingCensus,
};
use crate::{
    AssessmentGraphResourceBounds, CandidatesBinding, FeedForwardProgramSlot, MixerProgramSlot,
    ModelLoadPlan, PlannedMethod, ResourceCapacity, ResourceLimits, ResourcePlanner,
    TargetBlockProgramSlot,
};
use magnitude_cost_model::{
    Census, DraftCensus, DraftMethod, DraftWidthCensus, Kind, LaunchCounts, LayerCall, RoutedCall,
};
use magnitude_family_contracts::{ModelDefinition, SublayerIndex, WeightScope};
use magnitude_kernels as kernels;
use magnitude_state::{HistoryDomainLayout, KvCodec, LayerRef, ModelStateLayout};
use seismic::measure::pricing::{Bytes, Experts, Flops, Format, HistoryRead, Layout, Work};
use seismic::{BackendName, Element};

/// The words a banked plan's decode is encoded under, its streamed form:
/// neither whole-resident nor pre-resolved.
const STREAMED: &Words = &[
    ("policy", magnitude_kernels::constants::POLICY_WHOLE, false),
    ("policy", magnitude_kernels::constants::POLICY_RESOLVED, false),
    ("next_policy", magnitude_kernels::constants::POLICY_RESOLVED, false),
];

/// The words a banked plan's decode is encoded under when held whole-resident:
/// every layer is whole and resolved, so streaming-only launches are skipped.
const WHOLE: &Words = &[
    ("policy", magnitude_kernels::constants::POLICY_WHOLE, true),
    ("policy", magnitude_kernels::constants::POLICY_RESOLVED, true),
    ("next_policy", magnitude_kernels::constants::POLICY_RESOLVED, true),
];

/// An amount of one weight format's, summed per format.
trait PerFormat {
    /// What the amounts are, as an overflow names them.
    const WHAT: &'static str;
    fn format(&self) -> Format;
    fn amount(&mut self) -> &mut u64;
}

impl PerFormat for Bytes {
    const WHAT: &'static str = "weight bytes";
    fn format(&self) -> Format {
        self.format
    }
    fn amount(&mut self) -> &mut u64 {
        &mut self.bytes
    }
}

impl PerFormat for Flops {
    const WHAT: &'static str = "matmul FLOPs";
    fn format(&self) -> Format {
        self.format
    }
    fn amount(&mut self) -> &mut u64 {
        &mut self.flops
    }
}

/// `item` added to `into`'s entry of its format.
fn add<T: PerFormat>(into: &mut Vec<T>, mut item: T) -> Result<(), AssessmentError> {
    match into
        .iter_mut()
        .find(|entry| entry.format() == item.format())
    {
        Some(entry) => {
            let sum = entry
                .amount()
                .checked_add(*item.amount())
                .ok_or_else(|| demand_error(format!("{} overflow", T::WHAT)))?;
            *entry.amount() = sum;
        }
        None => into.push(item),
    }
    Ok(())
}

/// One call of the step: its entry, its work, and where it is a streamed
/// layer's call its gates price (its routed picks or its hint launch),
/// that layer's ([`LayerCall`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub launches: Launches,
    pub work: Work,
    pub layer: Option<LayerCall>,
}

/// One plain target decode step, call by call.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeDemand {
    pub calls: Vec<Call>,
    /// The submissions the step's calls end beyond its own: its streamed
    /// layers' parks and publications where the backend's parks are
    /// submission boundaries, each a submit and a wait beside device time.
    pub boundaries: u64,
}

impl DecodeDemand {
    /// The step's census ([`Census`]): each call's work, and its launches at
    /// each decode row count as its entry's declaration counts them, each
    /// entry's counted once; at a count its declaration gives no one count
    /// of, why, which prices only a launch of those rows. A streamed layer's
    /// routed calls carry its routed kind (`ExpertBankPlan::routed_kinds`),
    /// the numbering its routed passes are priced by.
    pub fn census(&self) -> Census {
        let mut counted = Vec::<(&Launches, LaunchCounts)>::new();
        let mut calls = Vec::with_capacity(self.calls.len());
        for call in &self.calls {
            let launches = match counted.iter().find(|(launches, _)| **launches == call.launches) {
                Some((_, counts)) => counts.clone(),
                None => {
                    let counts = LaunchCounts::new(
                        (1..=DECODE_ROWS)
                            .map(|rows| call.launches.at(rows).map_err(demand_reason))
                            .collect(),
                    );
                    counted.push((&call.launches, counts.clone()));
                    counts
                }
            };
            calls.push((
                Kind {
                    work: call.work.clone(),
                    launches,
                },
                call.layer,
            ));
        }
        Census::new(calls, self.boundaries, DECODE_ROWS)
    }
}

/// The rows of a decode step's row classes, beyond which a launch runs its
/// prefill forms (`VERIFY_ROWS`).
const DECODE_ROWS: u64 = kernels::constants::VERIFY_ROWS;

/// A weight format as the cost model keys it: its element's storage name.
pub fn format_of(element: Element) -> Format {
    Format(element.name())
}

/// The element a weight format names: its storage, registered under the
/// name [`format_of`] keys it by.
///
/// # Panics
/// Where no storage is registered under the name: every format is one
/// [`format_of`] named.
pub(crate) fn element_of(format: Format) -> Element {
    Element::named(format.0).unwrap_or_else(|| {
        panic!(
            "invariant violated: format {:?} names no registered storage",
            format.0
        )
    })
}

/// A history layout as the cost model keys it: its codec's identity.
pub fn layout_of(codec: KvCodec) -> Layout {
    Layout(codec.identity())
}

/// What a demand failure says, as the cost model names it.
fn demand_reason(error: AssessmentError) -> String {
    match error {
        AssessmentError::Demand(reason) => reason,
        other => other.to_string(),
    }
}

fn demand_error(message: impl Into<String>) -> AssessmentError {
    AssessmentError::Demand(message.into())
}

fn usize_bytes(bytes: usize) -> Result<u64, AssessmentError> {
    u64::try_from(bytes).map_err(|_| demand_error("state bytes exceed u64"))
}

fn product(left: u64, right: u64, what: &str) -> Result<u64, AssessmentError> {
    left.checked_mul(right)
        .ok_or_else(|| demand_error(format!("{what} overflow")))
}

/// The bytes of one row of `layer`'s component in `domain`, when it has one.
fn component_row_bytes(
    domain: &HistoryDomainLayout,
    layer: LayerRef,
) -> Result<Option<u64>, AssessmentError> {
    domain
        .components()
        .iter()
        .find(|component| component.layer == layer)
        .map(|component| {
            component.planes().iter().try_fold(0u64, |total, plane| {
                total
                    .checked_add(usize_bytes(plane.row_bytes)?)
                    .ok_or_else(|| demand_error("history row bytes overflow"))
            })
        })
        .transpose()
}

/// What one decode row of `layer`'s attention reads: its history's bytes per
/// token and, for a window domain, the most tokens it reads.
fn history_read(
    domains: &[HistoryDomainLayout],
    layer: LayerRef,
) -> Result<(u64, Option<u64>), AssessmentError> {
    for domain in domains {
        let window = match domain {
            HistoryDomainLayout::Token { .. } => None,
            HistoryDomainLayout::Window { rows, .. } => Some(usize_bytes(*rows)?),
            HistoryDomainLayout::Block { .. } => continue,
            HistoryDomainLayout::Shared { source, layers } => {
                if layers.contains(&layer) {
                    return history_read(domains, *source);
                }
                continue;
            }
        };
        if let Some(row_bytes) = component_row_bytes(domain, layer)? {
            return Ok((row_bytes, window));
        }
    }
    Err(demand_error(format!(
        "attention layer {layer:?} has no history"
    )))
}

fn block_component(
    domains: &[HistoryDomainLayout],
    layer: LayerRef,
) -> Result<(u64, u64), AssessmentError> {
    for domain in domains {
        if let HistoryDomainLayout::Block { rate, .. } = domain {
            if let Some(row_bytes) = component_row_bytes(domain, layer)? {
                let rate = u64::try_from(*rate)
                    .map_err(|_| demand_error("Block domain rate exceeds u64"))?;
                return Ok((row_bytes, rate));
            }
        }
    }
    Err(demand_error(format!("{layer:?} has no Block-domain rows")))
}

fn bank_bytes(
    layout: &ModelStateLayout,
    first: usize,
    count: usize,
    index: usize,
) -> Result<u64, AssessmentError> {
    layout
        .target_recurrent
        .get(first..first + count)
        .ok_or_else(|| demand_error(format!("block {index} has no recurrent state")))?
        .iter()
        .try_fold(0u64, |total, component| {
            total
                .checked_add(usize_bytes(component.bytes().map_err(demand_error)?)?)
                .ok_or_else(|| demand_error("recurrent state bytes overflow"))
        })
}

/// Per-block metadata needed to attach attention history, routed expert counts,
/// streamed layer identity, and recurrent state bytes to `SizingDraft`'s
/// recorded `SizedCall`s.
struct BlockContext<'a> {
    index: usize,
    slot: &'a TargetBlockProgramSlot,
    routed: Option<(u64, u64)>,
    streamed_layer: Option<(usize, magnitude_cost_model::RoutedKind)>,
    recurrent_first: usize,
    ngram_first: usize,
    layout: &'a ModelStateLayout,
}

fn block_routed_counts(definition: &ModelDefinition, index: usize) -> Option<(u64, u64)> {
    let block = definition.decoder.blocks.get(index)?;
    let sublayer = block.sublayers.get(1)?;
    let routed = crate::operators::routed::routed_ffn(&sublayer.op)?;
    Some((routed.experts, routed.selected))
}

fn call_from_sized(
    sized: &SizedCall,
    words: &'static Words,
    vocabulary: u64,
    block: Option<&BlockContext<'_>>,
    seen_gate_up_part: &mut bool,
) -> Result<Call, AssessmentError> {
    let launches = Launches::from_sized_call(sized, words);

    // A streamed gated layer runs `routed_gate_up_part` twice around
    // `routed_land`: the early pass owns the expert slab work and routed
    // layer tag; the late pass carries only its launch count.
    if sized.entry == "routed_gate_up_part" {
        if *seen_gate_up_part {
            return Ok(Call {
                launches,
                work: Work::default(),
                layer: None,
            });
        }
        *seen_gate_up_part = true;
    }

    let layer = block
        .and_then(|ctx| ctx.streamed_layer)
        .and_then(|(layer, kind)| match sized.entry {
            "routed_hint_route" | "routed_hint_select" => Some(LayerCall::Hint(layer)),
            "routed_gate_up_part"
            | "routed_up"
            | "routed_output"
            | "routed_output_project"
            | "routed_down"
            | "routed_down_project" => Some(LayerCall::Routed(RoutedCall { kind, layer })),
            _ => None,
        });

    let mut work = Work::default();
    let dim = |name: &str| -> Result<u64, AssessmentError> {
        sized
            .dimensions
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
            .ok_or_else(|| demand_error(format!("`{}` missing dimension `{name}`", sized.entry)))
    };

    // 1. Weights, expert slabs, and row lookups from tagged weight ports.
    let is_routed_block = block.and_then(|ctx| ctx.routed).is_some();
    let mut expert_slab_bytes = Vec::new();
    let mut expert_slab_flops = Vec::new();
    for port in sized.bound_ports.iter().filter(|port| port.is_weight) {
        if sized.entry == "embedding_rows" {
            if let Some(&rows) = port.extents.first().filter(|&&r| r > 0) {
                work.row_bytes = work
                    .row_bytes
                    .checked_add(port.bytes / rows)
                    .ok_or_else(|| demand_error("embedding row bytes overflow"))?;
            }
            continue;
        }
        if sized.entry == "stored_rows" {
            let width = *port.extents.last().unwrap_or(&0);
            let row = port
                .element
                .canonical_byte_len(&[1, width])
                .map_err(|error| demand_error(format!("stored_rows table: {error}")))?;
            let heads = block
                .and_then(|ctx| ctx.slot.ngram())
                .map_or(1, |binding| binding.heads());
            work.row_bytes = work
                .row_bytes
                .checked_add(product(row, heads, "stored row bytes")?)
                .ok_or_else(|| demand_error("stored row bytes overflow"))?;
            continue;
        }
        if sized.entry == "per_layer_inputs" {
            continue;
        }
        if port.extents.len() == 3 && is_routed_block {
            let [_, n, k] = port.extents[..] else {
                unreachable!()
            };
            let slab_bytes = port
                .element
                .canonical_byte_len(&[1, n, k])
                .map_err(|error| demand_error(format!("expert slab: {error}")))?;
            add(
                &mut expert_slab_bytes,
                Bytes {
                    format: format_of(port.element),
                    bytes: slab_bytes,
                },
            )?;
            add(
                &mut expert_slab_flops,
                Flops {
                    format: format_of(port.element),
                    flops: product(
                        2,
                        product(n, k, "expert slab elements")?,
                        "expert slab FLOPs",
                    )?,
                },
            )?;
        } else {
            add(
                &mut work.weights,
                Bytes {
                    format: format_of(port.element),
                    bytes: port.bytes,
                },
            )?;
            if port.extents.len() >= 2 {
                let elems = crate::planning::elements(&port.extents)
                    .ok_or_else(|| demand_error("weight matrix elements overflow"))?;
                add(
                    &mut work.row_flops,
                    Flops {
                        format: format_of(port.element),
                        flops: product(elems, 2, "matmul FLOPs")?,
                    },
                )?;
            }
        }
    }

    if !expert_slab_bytes.is_empty() {
        let (experts, selected) = block
            .and_then(|ctx| ctx.routed)
            .ok_or_else(|| {
                demand_error(format!(
                    "routed call `{}` has no routed block geometry",
                    sized.entry
                ))
            })?;
        work.experts = Some(Experts {
            experts,
            selected,
            slab_bytes: expert_slab_bytes,
            slab_flops: expert_slab_flops,
        });
    }

    // 2. Host-uploaded table conversions, per-layer entry, sampling, and stream row traffic.
    match sized.entry {
        "import_dense" | "repack_weight" => {
            if let Some(port) = sized.bound_ports.iter().find(|port| port.is_upload) {
                let k = dim("K")?;
                let n = dim("N")?;
                if n == 0 {
                    return Err(demand_error(format!(
                        "`{}` uploaded table port has zero rows",
                        sized.entry
                    )));
                }
                let row = port
                    .element
                    .canonical_byte_len(&[1, k])
                    .map_err(|error| demand_error(format!("uploaded table row: {error}")))?;
                work.row_bytes = work
                    .row_bytes
                    .checked_add(row)
                    .ok_or_else(|| demand_error("uploaded table row bytes overflow"))?;
                work.upload_row_bytes = work
                    .upload_row_bytes
                    .checked_add(port.bytes / n)
                    .ok_or_else(|| demand_error("upload row bytes overflow"))?;
            } else if sized.entry == "import_dense" && block.is_none() {
                let k = dim("K")?;
                work.row_bytes = work
                    .row_bytes
                    .checked_add(product(k, 4, "import dense row bytes")?)
                    .ok_or_else(|| demand_error("import dense row bytes overflow"))?;
            }
        }
        "per_layer_inputs" => {
            let channels = product(dim("L")?, dim("P")?, "per-layer channels")?;
            work.row_bytes = work
                .row_bytes
                .checked_add(product(channels, 4, "per-layer inputs row bytes")?)
                .ok_or_else(|| demand_error("per-layer inputs row bytes overflow"))?;
        }
        "conditioning_overlay" => {
            let d = dim("D")?;
            work.row_bytes = work
                .row_bytes
                .checked_add(product(d, 4, "conditioning overlay row bytes")?)
                .ok_or_else(|| demand_error("conditioning overlay row bytes overflow"))?;
        }
        "sample_rows" => {
            work.row_bytes = work
                .row_bytes
                .checked_add(product(vocabulary, 4, "logits bytes")?)
                .ok_or_else(|| demand_error("logits bytes overflow"))?;
        }
        "stream_collapse" => {
            let s = dim("S")?;
            let h = dim("H")?;
            let sh = product(s, h, "stream collapse elements")?;
            let elems = sh
                .checked_add(h)
                .ok_or_else(|| demand_error("stream collapse elements overflow"))?;
            work.row_bytes = work
                .row_bytes
                .checked_add(product(elems, 4, "stream collapse bytes")?)
                .ok_or_else(|| demand_error("stream collapse bytes overflow"))?;
        }
        "stream_join" => {
            let s = dim("SN")?;
            let h = dim("H")?;
            let sh = product(s, h, "stream join elements")?;
            let two_sh = product(2, sh, "stream join elements")?;
            let elems = two_sh
                .checked_add(h)
                .ok_or_else(|| demand_error("stream join elements overflow"))?;
            work.row_bytes = work
                .row_bytes
                .checked_add(product(elems, 4, "stream join bytes")?)
                .ok_or_else(|| demand_error("stream join bytes overflow"))?;
        }
        _ => {}
    }

    // 3. Per-block history reads and recurrent state advances.
    if let Some(ctx) = block {
        match sized.entry {
            "attention_decode" | "attention_prefill" | "attention_decode_k8v4" => {
                if let MixerProgramSlot::Attention(binding) = ctx.slot.mixer() {
                    let layer = u32::try_from(ctx.index)
                        .map_err(|_| demand_error("block index exceeds u32"))?;
                    let (row_bytes, window) =
                        history_read(&ctx.layout.target_history, LayerRef::Target(layer))?;
                    let widths = binding.shape.widths;
                    let flops_per_token = 2 * widths.kv_heads * widths.group * (2 * widths.width);
                    let history_layout = if sized.entry == "attention_decode_k8v4" {
                        KvCodec::AffineK8V4
                    } else {
                        KvCodec::Dense
                    };
                    let window = match binding.selection {
                        None => window,
                        Some(sel) => {
                            let selected = product(
                                sel.indexer.top + 1,
                                sel.shape.block,
                                "selected tokens",
                            )?;
                            Some(window.map_or(selected, |w| w.min(selected)))
                        }
                    };
                    work.history = Some(HistoryRead {
                        layout: Some(layout_of(history_layout)),
                        bytes_per_token: row_bytes,
                        flops_per_token,
                        window,
                    });
                }
            }
            "indexer_score" | "rows_topk" | "candidate_blocks" => {
                if let MixerProgramSlot::Attention(binding) = ctx.slot.mixer() {
                    if let Some(sel) = binding.selection {
                        let layer = u32::try_from(ctx.index)
                            .map_err(|_| demand_error("block index exceeds u32"))?;
                        let keys = LayerRef::IndexKeys {
                            block: layer,
                            part: 0,
                        };
                        let (key_row, rate) =
                            block_component(&ctx.layout.target_history, keys)?;
                        let window = match sel.indexer.candidates {
                            CandidatesBinding::All | CandidatesBinding::Produce(_) => None,
                            CandidatesBinding::Restrict(pool) => {
                                Some(product(pool.units(), rate, "candidate pool tokens")?)
                            }
                        };
                        work.history = match sized.entry {
                            "indexer_score" => {
                                let score = product(
                                    2 * sel.indexer.heads,
                                    sel.indexer.width,
                                    "indexer score FLOPs",
                                )?;
                                Some(HistoryRead {
                                    layout: None,
                                    bytes_per_token: (key_row + 4).div_ceil(rate),
                                    flops_per_token: score.div_ceil(rate),
                                    window,
                                })
                            }
                            "rows_topk" => Some(HistoryRead {
                                layout: None,
                                bytes_per_token: 4u64.div_ceil(rate),
                                flops_per_token: 0,
                                window,
                            }),
                            "candidate_blocks" => {
                                if let CandidatesBinding::Produce(pool) = sel.indexer.candidates {
                                    let block_tokens =
                                        product(pool.block_units, rate, "candidate block tokens")?;
                                    let block_bytes =
                                        product(4, pool.block_units, "candidate block bytes")?
                                            + 4
                                            + 8;
                                    Some(HistoryRead {
                                        layout: None,
                                        bytes_per_token: block_bytes.div_ceil(block_tokens),
                                        flops_per_token: 0,
                                        window: None,
                                    })
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                    }
                }
            }
            "state_space_step"
            | "gated_delta_step"
            | "gated_delta_step_convolved"
            | "kimi_delta_step" => {
                work.sequence_bytes = work
                    .sequence_bytes
                    .checked_add(bank_bytes(ctx.layout, ctx.recurrent_first, 3, ctx.index)?)
                    .ok_or_else(|| demand_error("sequence state bytes overflow"))?;
            }
            "short_conv_rows" => {
                work.sequence_bytes = work
                    .sequence_bytes
                    .checked_add(bank_bytes(ctx.layout, ctx.recurrent_first, 1, ctx.index)?)
                    .ok_or_else(|| demand_error("sequence state bytes overflow"))?;
            }
            "ngram_rows" => {
                if let Some(binding) = ctx.slot.ngram() {
                    let token_window = ctx.ngram_first + binding.bank_components() - 1;
                    work.sequence_bytes = work
                        .sequence_bytes
                        .checked_add(bank_bytes(ctx.layout, token_window, 1, ctx.index)?)
                        .ok_or_else(|| demand_error("sequence state bytes overflow"))?;
                }
            }
            "stream_mix_sinkhorn" => {
                if let Some(BlockStreams::Sinkhorn(binding)) = ctx.slot.block_streams() {
                    let SinkhornBinding {
                        hidden,
                        streams,
                        mixing,
                        ..
                    } = binding;
                    let sh = product(streams, hidden, "sinkhorn stream elements")?;
                    let ss = product(streams, streams, "sinkhorn comb elements")?;
                    let pre = product(binding.carried(), streams, "sinkhorn pre elements")?;
                    let elems = [sh, hidden, pre, streams, ss]
                        .into_iter()
                        .try_fold(0u64, |acc, v| {
                            acc.checked_add(v)
                                .ok_or_else(|| demand_error("sinkhorn row elements overflow"))
                        })?;
                    work.row_bytes = work
                        .row_bytes
                        .checked_add(product(elems, 4, "sinkhorn row bytes")?)
                        .ok_or_else(|| demand_error("sinkhorn row bytes overflow"))?;
                    let normalizations = if mixing.iterations == 0 {
                        0
                    } else {
                        product(2, u64::from(mixing.iterations), "sinkhorn normalizations")? - 1
                    };
                    let flops = product(
                        product(normalizations, 2, "sinkhorn flops")?,
                        ss,
                        "sinkhorn flops",
                    )?;
                    add(
                        &mut work.row_flops,
                        Flops {
                            format: format_of(Element::f32()),
                            flops,
                        },
                    )?;
                }
            }
            _ => {}
        }
    }

    Ok(Call {
        launches,
        work,
        layer,
    })
}

impl DecodeDemand {
    /// Derive one plain decode step's demand directly from `SizingDraft`'s
    /// recorded decode-regime census in `graph`.
    ///
    /// When `resident` is `true`, blocks that stream routed experts use their
    /// resident-form decode calls (`RunStage::Resident`) instead of their
    /// streamed-form calls (`RunStage::Whole`).
    pub(crate) fn from_census(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
        graph: &AssessmentGraphResourceBounds,
        resident: bool,
    ) -> Result<Self, AssessmentError> {
        Self::from_family_censuses(
            definition,
            load,
            codec,
            &graph.target.census,
            &graph.readout_census,
            resident,
        )
    }

    pub(crate) fn from_family_censuses(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
        target_census: &TargetFamilySizingCensus,
        readout_census: &ReadoutSizingCensus,
        resident: bool,
    ) -> Result<Self, AssessmentError> {
        let program = load
            .program_plan(definition, codec)
            .map_err(|error| demand_error(error.to_string()))?;
        let layout =
            ModelStateLayout::derive(&definition.decoder, &[], codec, 0).map_err(demand_error)?;
        let words: &'static Words = if load.expert_bank().is_some() {
            if resident { WHOLE } else { STREAMED }
        } else {
            &[]
        };
        let vocabulary = definition.decoder.vocabulary;
        let backend = target_census
            .entry_calls
            .first()
            .or_else(|| readout_census.decode_calls.first())
            .map(|c| c.backend)
            .ok_or_else(|| demand_error("sizing census recorded no decode calls"))?;

        let mut calls = Vec::new();
        let mut timelines = target_census.entry_timelines;
        let mut unused_gate_up_part = false;

        for sized in &target_census.entry_calls {
            calls.push(call_from_sized(
                sized,
                words,
                vocabulary,
                None,
                &mut unused_gate_up_part,
            )?);
        }

        let target_blocks = program.target().blocks();
        let mut recurrent_component = 0usize;
        for (index, slot) in target_blocks.iter().enumerate() {
            let mixer_components = match slot.mixer() {
                MixerProgramSlot::Attention(_) => 0,
                MixerProgramSlot::StateSpace(_) | MixerProgramSlot::Recurrent(_) => 3,
                MixerProgramSlot::ShortConv(_) => 1,
            };
            let ngram_first = recurrent_component + mixer_components;
            let ngram_components = slot.ngram().map_or(0, |b| b.bank_components());
            let streamed_layer = if resident {
                None
            } else {
                let block = u32::try_from(index)
                    .map_err(|_| demand_error("block index exceeds u32"))?;
                let ffn_sublayer = SublayerIndex { block, sublayer: 1 };
                let routed_scope = match slot.feed_forward() {
                    Some(FeedForwardProgramSlot::Parallel(_)) => {
                        DenseBesideRouted::scopes(ffn_sublayer)[1]
                    }
                    _ => WeightScope::TargetSublayer(ffn_sublayer),
                };
                streamed::served(load, routed_scope)
                    .map(|(bank, layer)| (layer, bank.routed_kinds()[layer]))
            };
            let ctx = BlockContext {
                index,
                slot,
                routed: block_routed_counts(definition, index),
                streamed_layer,
                recurrent_first: recurrent_component,
                ngram_first,
                layout: &layout,
            };
            let shape_idx = *target_census
                .block_map
                .get(index)
                .ok_or_else(|| demand_error(format!("block {index} absent from sizing census")))?;
            let shape = &target_census.block_shapes[shape_idx];
            let (block_calls, block_timelines) = if resident {
                shape
                    .resident
                    .as_ref()
                    .map_or((shape.calls.as_slice(), shape.timelines), |(c, t)| {
                        (c.as_slice(), *t)
                    })
            } else {
                (shape.calls.as_slice(), shape.timelines)
            };
            timelines += block_timelines;
            let mut seen_gate_up_part = false;
            for sized in block_calls {
                calls.push(call_from_sized(
                    sized,
                    words,
                    vocabulary,
                    Some(&ctx),
                    &mut seen_gate_up_part,
                )?);
            }
            recurrent_component = ngram_first + ngram_components;
        }

        for sized in &readout_census.decode_calls {
            calls.push(call_from_sized(
                sized,
                words,
                vocabulary,
                None,
                &mut unused_gate_up_part,
            )?);
        }

        let boundaries = if !backend.parks_in_kernels() {
            timelines as u64
        } else {
            0
        };

        Ok(Self { calls, boundaries })
    }
}

/// One candidate draft width's decode demand.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftWidthDemand {
    pub proposals: u8,
    pub rows_per_sequence: u64,
    pub demand: DecodeDemand,
}

/// Demand of a load's drafter across its certified proposal widths.
#[derive(Clone, Debug, PartialEq)]
pub struct DraftDemand {
    pub method: DraftMethod,
    pub widths: Vec<DraftWidthDemand>,
    pub max_proposals: u8,
}

impl DraftDemand {
    /// Convert this drafter's demand into a cost-model [`DraftCensus`].
    pub fn census(&self) -> DraftCensus {
        DraftCensus {
            method: self.method,
            widths: self
                .widths
                .iter()
                .map(|w| DraftWidthCensus {
                    proposals: w.proposals,
                    rows_per_sequence: w.rows_per_sequence,
                    census: w.demand.census(),
                })
                .collect(),
            max_proposals: self.max_proposals,
        }
    }

    /// Derive drafter demand directly from the `DraftFamilySizingCensus`
    /// already retained in `graph` during `StructuralCharge::of` (zero extra
    /// outlining cost). Returns `Ok(None)` when `method` does not draft or no
    /// proposing widths were recorded.
    pub(crate) fn from_census(
        definition: &ModelDefinition,
        method: PlannedMethod,
        graph: &AssessmentGraphResourceBounds,
    ) -> Result<Option<Self>, AssessmentError> {
        let Some(draft_census) = &graph.draft_census else {
            return Ok(None);
        };
        Self::from_family_census(definition, method, draft_census)
    }

    pub(crate) fn from_family_census(
        definition: &ModelDefinition,
        method: PlannedMethod,
        draft_census: &DraftFamilySizingCensus,
    ) -> Result<Option<Self>, AssessmentError> {
        let draft_method = match method {
            PlannedMethod::Plain => return Ok(None),
            PlannedMethod::Mtp { .. } => DraftMethod::Mtp,
            PlannedMethod::DFlash { .. } => DraftMethod::Block,
        };
        if draft_census.widths.is_empty() {
            return Ok(None);
        }
        let vocabulary = definition.decoder.vocabulary;
        let mut widths = Vec::with_capacity(draft_census.widths.len());
        for width in &draft_census.widths {
            if width.calls.is_empty() {
                continue;
            }
            let backend = width.calls[0].backend;
            let mut calls = Vec::with_capacity(width.calls.len());
            let mut unused_gate_up_part = false;
            for sized in &width.calls {
                calls.push(call_from_sized(
                    sized,
                    &[],
                    vocabulary,
                    None,
                    &mut unused_gate_up_part,
                )?);
            }
            let boundaries = if !backend.parks_in_kernels() {
                width.timelines as u64
            } else {
                0
            };
            widths.push(DraftWidthDemand {
                proposals: width.proposals,
                rows_per_sequence: width.rows_per_sequence,
                demand: DecodeDemand { calls, boundaries },
            });
        }
        if widths.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            method: draft_method,
            widths,
            max_proposals: method.widest(),
        }))
    }
}

/// Outline only the drafter family via `SizingDraft` and build `DraftDemand`
/// for load-time `LaunchPrices` (returns `Ok(None)` when `method` does not
/// draft).
pub(crate) fn draft_demand_direct(
    definition: &ModelDefinition,
    load: &ModelLoadPlan,
    codec: KvCodec,
    method: PlannedMethod,
    limits: ResourceLimits,
    capacity: ResourceCapacity,
    backend: BackendName,
) -> Result<Option<DraftDemand>, AssessmentError> {
    if !method.drafts() {
        return Ok(None);
    }
    let plan = AssessmentGraphResourceBounds::program_plan(definition, load, method, codec)
        .map_err(AssessmentError::Graph)?;
    let state = ResourcePlanner::state_plan(definition, load, method, codec, limits, capacity)
        .map_err(AssessmentError::Plan)?;
    let Some(draft_census) = AssessmentGraphResourceBounds::drafter_census_direct(
        definition, load, &plan, &state, method, limits, backend,
    )
    .map_err(AssessmentError::Graph)?
    else {
        return Ok(None);
    };
    DraftDemand::from_family_census(definition, method, &draft_census)
}

/// Outline only the target and readout families via `SizingDraft` and build
/// `DecodeDemand` without constructing a full `StructuralCharge` or calling
/// `Declarations::of_load`.
pub(crate) fn decode_demand_direct(
    definition: &ModelDefinition,
    load: &ModelLoadPlan,
    codec: KvCodec,
    method: PlannedMethod,
    limits: ResourceLimits,
    capacity: ResourceCapacity,
    backend: BackendName,
    resident: bool,
) -> Result<DecodeDemand, AssessmentError> {
    let plan = AssessmentGraphResourceBounds::program_plan(definition, load, method, codec)
        .map_err(AssessmentError::Graph)?;
    let state = ResourcePlanner::state_plan(definition, load, method, codec, limits, capacity)
        .map_err(AssessmentError::Plan)?;
    let target = crate::programs::native_target_graph::checked_target_family(
        backend,
        load,
        &definition.decoder,
        &state,
        plan.target(),
        limits,
        load.pass(limits),
    )
    .map_err(AssessmentError::Graph)?;
    let (_, _, readout_census) =
        crate::programs::graph::readout::checked_readout_storage_with_census(
            backend,
            load,
            &definition.decoder,
            limits,
        )
        .map_err(AssessmentError::Graph)?;
    DecodeDemand::from_family_censuses(
        definition,
        load,
        codec,
        &target.census,
        &readout_census,
        resident,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::assessment::launches::Declarations;

    /// The declarations of a load of `definition` under `codec` on
    /// `backend`, from its kernel inventory (kept for synthetic unit tests in
    /// `estimate::tests`).
    pub(crate) fn declarations(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
        backend: BackendName,
    ) -> Result<Declarations, AssessmentError> {
        let program = load
            .program_plan(definition, codec)
            .map_err(|error| demand_error(error.to_string()))?;
        Declarations::of_load(
            backend,
            &program,
            definition,
            load,
            crate::native::TuningLimits::of(crate::assessment::assess::tests::LIMITS, load, definition),
        )
    }

    /// The decode demand of `definition` under `codec` on `backend`, derived
    /// directly via `SizingDraft`.
    pub(crate) fn demand_on(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
        backend: BackendName,
    ) -> Result<DecodeDemand, AssessmentError> {
        let limits = crate::assessment::assess::tests::LIMITS;
        let capacity = ResourceCapacity {
            domain_bytes: 64 * 1024 * 1024 * 1024,
            tensor_operations: crate::TensorOperations::of(false),
            allocations: seismic::NativeAllocations::EXACT,
        };
        decode_demand_direct(
            definition,
            load,
            codec,
            PlannedMethod::Plain,
            limits,
            capacity,
            backend,
            false,
        )
    }

    /// The decode demand of `definition` under `codec` on the CPU backend.
    pub(crate) fn demand(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
    ) -> Result<DecodeDemand, AssessmentError> {
        demand_on(definition, load, codec, BackendName::Cpu)
    }

    /// The bytes one row of one sequence streams, weights and expert slabs
    /// included, beside history.
    pub(crate) fn streamed(demand: &DecodeDemand) -> u64 {
        demand
            .calls
            .iter()
            .map(|call| {
                let work = &call.work;
                let bytes = |bytes: &[Bytes]| bytes.iter().map(|bytes| bytes.bytes).sum::<u64>();
                bytes(&work.weights)
                    + work.row_bytes
                    + work.sequence_bytes
                    + work
                        .experts
                        .as_ref()
                        .map_or(0, |experts| experts.selected * bytes(&experts.slab_bytes))
            })
            .sum()
    }

    #[test]
    fn census_decode_demand_runs_on_every_backend() {
        let definition = crate::planning::fixtures::fixture_definition();
        let manifest = crate::planning::fixtures::fixture_manifest(&definition);
        for backend in BackendName::ALL {
            let load = ModelLoadPlan::derive(
                &manifest,
                &definition,
                crate::ComponentSelection {
                    head: false,
                    vision: false,
                },
                crate::resident_layout(crate::ExecutionPath::Native, backend),
                PlannedMethod::Plain,
                crate::assessment::assess::tests::LIMITS,
            )
            .unwrap();
            let demand = demand_on(&definition, &load, KvCodec::Dense, backend).unwrap();
            assert!(!demand.calls.is_empty(), "{backend:?}");
            assert!(streamed(&demand) > 0, "{backend:?}");
            for call in &demand.calls {
                for rows in [1, magnitude_kernels::constants::VERIFY_ROWS] {
                    call.launches.at(rows).unwrap_or_else(|error| {
                        panic!(
                            "{backend:?} `{}` at {rows} rows: {error}",
                            call.launches.kernel().name()
                        )
                    });
                }
            }
        }
    }
}
