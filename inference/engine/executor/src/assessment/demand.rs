//! Header-derived decode demand: one record per call of a plain target
//! decode step, in the order the step runs them. Each names the entry it
//! runs ([`Launches`], whose launches at a row count are the backend's
//! declaration's) and the work it does: the weights it reads once per step,
//! the bytes and FLOPs of each row, the state of each sequence, and the
//! history it reads per token of context. Derived from the same program and
//! load plans native preparation uses, and from the state layout the memory
//! terms use for state and history bytes.
//!
//! Matmul FLOPs are 2 × the weight's logical elements per row (a multiply
//! and an add per weight), of a routed tensor the selected experts' slabs.
//! Attention FLOPs per row and token read are 2 · KV · G · (W_k + W_v): each
//! query head scores a key and accumulates a value of its head width.
//! Recurrent and state-space state advances are priced by their bytes only:
//! their arithmetic per state element is the kernels', which no plan
//! states.

use super::launches::{operand_recipe, Declarations, Kernel, Launches};
use super::AssessmentError;
use crate::operators::attention::graph::selected_segments;
use crate::operators::gated_delta::graph::StepForm;
use crate::operators::routed::fused_graph::DecodeForm;
use crate::operators::streams::{MixSite, StreamsBinding};
use crate::operators::{parallel::DenseBesideRouted, InputForm};
use crate::{
    FeedForwardProgramSlot, GeneralRoutedBinding, MixerProgramSlot, ModelLoadPlan,
    PerLayerEntryBinding, ReadoutHead, SublayerTail, WeightPlan,
};
use magnitude_family_contracts::{
    InputNorm, ModelDefinition, SublayerIndex, WeightKind, WeightScope,
};
use magnitude_kernels as kernels;
use magnitude_state::{HistoryDomainLayout, HistoryEncoding, KvCodec, LayerRef, ModelStateLayout};
use seismic::{Element, OperandRecipe};
use std::collections::HashMap;

/// History one call reads for each token of context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryRead {
    /// The layout an attention entry reads it in; none for an indexer's
    /// F32 block keys and scores, which a selection call streams.
    pub layout: Option<KvCodec>,
    pub bytes_per_token: u64,
    /// FLOPs of one row per token read: attention's bf16 score and value
    /// products.
    pub flops_per_token: u64,
    /// The most tokens read: a window domain's window, or the blocks a
    /// selected read reads and its tail. `None` reads every token.
    pub window: Option<u64>,
}

impl HistoryRead {
    /// The tokens read at `depth` tokens of context.
    pub fn tokens_at(&self, depth: u32) -> u64 {
        self.window
            .map_or(u64::from(depth), |window| window.min(u64::from(depth)))
    }
}

/// Matmul FLOPs over weights resident in one format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flops {
    pub format: Element,
    pub flops: u64,
}

/// Weight bytes resident in one format, which its own weight rate prices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bytes {
    pub format: Element,
    pub bytes: u64,
}

/// An amount of one weight format's, summed per format.
trait PerFormat {
    /// What the amounts are, as an overflow names them.
    const WHAT: &'static str;
    fn format(&self) -> Element;
    fn amount(&mut self) -> &mut u64;
}

impl PerFormat for Bytes {
    const WHAT: &'static str = "weight bytes";
    fn format(&self) -> Element {
        self.format
    }
    fn amount(&mut self) -> &mut u64 {
        &mut self.bytes
    }
}

impl PerFormat for Flops {
    const WHAT: &'static str = "matmul FLOPs";
    fn format(&self) -> Element {
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

/// The bytes of `weights`, by format.
fn weight_bytes(weights: &[&WeightPlan]) -> Result<Vec<Bytes>, AssessmentError> {
    let mut bytes = Vec::new();
    for weight in weights {
        add(
            &mut bytes,
            Bytes {
                format: weight.resident,
                bytes: weight.resident_bytes,
            },
        )?;
    }
    Ok(bytes)
}

/// The expert slabs a routed call reads: `selected` of `experts` equal
/// slabs per row, the union of the rows' choices once per step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Experts {
    pub experts: u64,
    pub selected: u64,
    /// One expert's bytes over every expert tensor the call reads, by
    /// format.
    pub slab_bytes: Vec<Bytes>,
    /// One expert's FLOPs per row over those tensors, by format.
    pub slab_flops: Vec<Flops>,
}

/// What one call reads and computes. Resident weights and per-sequence
/// state are in device storage, as the load plan places them; the host
/// writes upload rows (`Tensor::shared`, upload storage).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Work {
    /// Weights read whole, once per step, by format.
    pub weights: Vec<Bytes>,
    pub experts: Option<Experts>,
    /// Device bytes each row reads and writes.
    pub row_bytes: u64,
    /// Upload bytes each row reads.
    pub upload_row_bytes: u64,
    /// Each row's matmul FLOPs, by weight format.
    pub row_flops: Vec<Flops>,
    /// Device state each sequence reads and writes, whatever its depth.
    pub sequence_bytes: u64,
    pub history: Option<HistoryRead>,
}

/// One call of the step: its entry and its work.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub launches: Launches,
    pub work: Work,
}

/// One plain target decode step, call by call.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeDemand {
    pub calls: Vec<Call>,
}

fn demand_error(message: impl Into<String>) -> AssessmentError {
    AssessmentError::Demand(message.into())
}

/// The logical elements of a weight.
fn elements(weight: &WeightPlan) -> Result<u64, AssessmentError> {
    crate::planning::elements(&weight.shape).ok_or_else(|| {
        demand_error(format!(
            "weight {:?} elements overflow",
            weight.descriptor.name
        ))
    })
}

/// The FLOPs of one row through `weight`, a matrix: a multiply and an add
/// per logical element. A vector (a norm, a gate's scale) is read, not
/// multiplied through: none.
fn matmul_flops(weight: &WeightPlan) -> Result<Option<Flops>, AssessmentError> {
    if weight.shape.len() < 2 {
        return Ok(None);
    }
    Ok(Some(Flops {
        format: weight.resident,
        flops: elements(weight)?
            .checked_mul(2)
            .ok_or_else(|| demand_error("matmul FLOPs overflow"))?,
    }))
}

/// The work of projecting each row through every weight in `weights`.
fn projection(weights: &[&WeightPlan]) -> Result<Work, AssessmentError> {
    let mut row_flops = Vec::new();
    for weight in weights {
        if let Some(flops) = matmul_flops(weight)? {
            add(&mut row_flops, flops)?;
        }
    }
    Ok(Work {
        weights: weight_bytes(weights)?,
        row_flops,
        ..Work::default()
    })
}

/// Weights read whole without a matmul (norms, routers' biases, taps).
fn weights(weights: &[&WeightPlan]) -> Result<Work, AssessmentError> {
    Ok(Work {
        weights: weight_bytes(weights)?,
        ..Work::default()
    })
}

fn rows(bytes: u64) -> Work {
    Work {
        row_bytes: bytes,
        ..Work::default()
    }
}

fn sequences(bytes: u64) -> Work {
    Work {
        sequence_bytes: bytes,
        ..Work::default()
    }
}

/// One expert's bytes and FLOPs per row of an expert tensor of `experts`
/// equal slabs.
fn slab(weight: &WeightPlan, experts: u64) -> Result<(Bytes, Flops), AssessmentError> {
    let elements = elements(weight)?;
    if experts == 0 || weight.resident_bytes % experts != 0 || elements % experts != 0 {
        return Err(demand_error(format!(
            "expert tensor {:?} does not split into {experts} slabs",
            weight.descriptor.name
        )));
    }
    Ok((
        Bytes {
            format: weight.resident,
            bytes: weight.resident_bytes / experts,
        },
        Flops {
            format: weight.resident,
            flops: 2 * (elements / experts),
        },
    ))
}

/// The slabs of `tensors`, `selected` of `experts` per row.
fn experts(
    tensors: &[&WeightPlan],
    experts: u64,
    selected: u64,
) -> Result<Experts, AssessmentError> {
    if selected > experts {
        return Err(demand_error(format!(
            "{selected} selected of {experts} experts"
        )));
    }
    let slabs = tensors
        .iter()
        .map(|tensor| slab(tensor, experts))
        .collect::<Result<Vec<_>, _>>()?;
    let mut slab_bytes = Vec::new();
    let mut slab_flops = Vec::new();
    for (bytes, flops) in &slabs {
        add(&mut slab_bytes, *bytes)?;
        add(&mut slab_flops, *flops)?;
    }
    Ok(Experts {
        experts,
        selected,
        slab_bytes,
        slab_flops,
    })
}

/// Work done by one call: the sum of each part's.
fn join(parts: impl IntoIterator<Item = Work>) -> Result<Work, AssessmentError> {
    let mut joined = Work::default();
    for part in parts {
        let sum = |left: u64, right: u64, what: &str| {
            left.checked_add(right)
                .ok_or_else(|| demand_error(format!("call {what} overflow")))
        };
        for bytes in part.weights {
            add(&mut joined.weights, bytes)?;
        }
        joined.row_bytes = sum(joined.row_bytes, part.row_bytes, "row bytes")?;
        joined.upload_row_bytes = sum(
            joined.upload_row_bytes,
            part.upload_row_bytes,
            "upload bytes",
        )?;
        for flops in part.row_flops {
            add(&mut joined.row_flops, flops)?;
        }
        joined.sequence_bytes = sum(joined.sequence_bytes, part.sequence_bytes, "state bytes")?;
        joined.experts = match (joined.experts.take(), part.experts) {
            (None, experts) | (experts, None) => experts,
            (Some(_), Some(_)) => return Err(demand_error("a call reads two expert sets")),
        };
        joined.history = match (joined.history, part.history) {
            (None, history) | (history, None) => history,
            (Some(_), Some(_)) => return Err(demand_error("a call reads two histories")),
        };
    }
    Ok(joined)
}

/// The step as it accumulates.
struct Step<'a> {
    declarations: &'a Declarations,
    /// Each entry's launches at each operand recipe, resolved once.
    resolved: HashMap<(&'static str, OperandRecipe), Launches>,
    calls: Vec<Call>,
}

/// The operand recipe of the weights `work` reads, whole or as experts.
fn work_recipe(work: &Work) -> OperandRecipe {
    let experts = work.experts.iter().flat_map(|experts| &experts.slab_bytes);
    operand_recipe(work.weights.iter().chain(experts).map(|bytes| bytes.format))
}

impl Step<'_> {
    /// One call of `kernel` doing `work`.
    fn call(&mut self, kernel: Kernel, work: Work) -> Result<(), AssessmentError> {
        let key = (kernel.name(), work_recipe(&work));
        let launches = match self.resolved.get(&key) {
            Some(launches) => launches.clone(),
            None => {
                let launches = self.declarations.launches(kernel, key.1, &[])?;
                self.resolved.insert(key, launches.clone());
                launches
            }
        };
        self.calls.push(Call { launches, work });
        Ok(())
    }

    /// One call of `kernel` stating `dimensions` its launch clauses read.
    fn call_with(
        &mut self,
        kernel: Kernel,
        dimensions: &[(&'static str, u64)],
        work: Work,
    ) -> Result<(), AssessmentError> {
        let launches = self.declarations.launches(kernel, work_recipe(&work), dimensions)?;
        self.calls.push(Call { launches, work });
        Ok(())
    }
}

/// The planned weight of `kind` in `scope`, checked against the element the
/// program slot binds.
fn planned<'a>(
    load: &'a ModelLoadPlan,
    scope: WeightScope,
    kind: WeightKind,
    bound: Element,
) -> Result<&'a WeightPlan, AssessmentError> {
    let weight = load
        .target()
        .iter()
        .find(|weight| weight.role.scope == scope && weight.role.kind == kind)
        // An output projection of a scope stored as planes is planned and
        // not placed: its canonical full-projection demand stands.
        .or_else(|| {
            (kind == WeightKind::Output)
                .then(|| load.planned_output(scope))
                .flatten()
        })
        .ok_or_else(|| demand_error(format!("{kind:?} weight is absent for {scope:?}")))?;
    if weight.resident != bound {
        return Err(demand_error(format!(
            "{kind:?} binding for {scope:?} disagrees with the resident load plan"
        )));
    }
    Ok(weight)
}

/// A sublayer's output by its tail: the operator's own output entry
/// (`residual`, which joins the residual), or the `output` projection into
/// F32 rows and the post-norm row op, whose norm weight `planned` checks.
fn tail<'a>(
    step: &mut Step<'_>,
    tail: SublayerTail,
    residual: Kernel,
    output: &WeightPlan,
    planned: impl Fn(WeightKind, Element) -> Result<&'a WeightPlan, AssessmentError>,
) -> Result<(), AssessmentError> {
    match tail {
        SublayerTail::Residual => step.call(residual, projection(&[output])?),
        SublayerTail::PostNorm { norm, .. } => {
            post_norm(step, output, planned(WeightKind::PostNorm, norm)?)
        }
    }
}

/// A post-norm tail (`operators::output`): the `output` projection into F32
/// rows (`project_rows`), then the row op normalizing it with `norm` into
/// the residual (`post_norm_residual`).
fn post_norm(
    step: &mut Step<'_>,
    output: &WeightPlan,
    norm: &WeightPlan,
) -> Result<(), AssessmentError> {
    step.call(
        Kernel::of::<kernels::project_rows::Entry>(),
        projection(&[output])?,
    )?;
    step.call(
        Kernel::of::<kernels::post_norm_residual::Entry>(),
        weights(&[norm])?,
    )
}

/// The bytes of `elements` values of `element`.
fn row_bytes(element: Element, elements: u64) -> Result<u64, AssessmentError> {
    element
        .canonical_byte_len(&[elements])
        .map_err(|error| demand_error(format!("{} row of {elements}: {error}", element.name())))
}

/// `operators::per_layer::graph::per_layer_entry`, once per step after the
/// embedding: the embedded row rounded (`import_dense`) and projected to
/// every layer's channels (`project_rows`), the row's host-table row, which
/// the host uploads, converted to its resident representation
/// (`import_dense` or `repack_weight`), both combined (`per_layer_inputs`),
/// and the result copied into the program's per-layer rows
/// (`conditioning_overlay`).
fn per_layer_entry(
    step: &mut Step<'_>,
    load: &ModelLoadPlan,
    entry: PerLayerEntryBinding,
) -> Result<(), AssessmentError> {
    let channels = entry
        .layers
        .checked_mul(entry.width)
        .ok_or_else(|| demand_error("per-layer channels overflow"))?;
    let table = load
        .host_tables()
        .iter()
        .find(|table| table.role.kind == WeightKind::PerLayerTable)
        .ok_or_else(|| demand_error("a per-layer entry without its host table"))?;
    let table_rows = table.shape[0];
    if table.source != entry.table_source || table_rows == 0 || table.bytes % table_rows != 0 {
        return Err(demand_error(format!(
            "host table {:?} disagrees with the per-layer entry",
            table.descriptor.name
        )));
    }
    step.call(
        Kernel::of::<kernels::import_dense::Entry>(),
        rows(row_bytes(Element::f32(), entry.hidden)?),
    )?;
    let projected = planned(
        load,
        WeightScope::Target,
        WeightKind::PerLayerModelProjection,
        entry.projection,
    )?;
    step.call(
        Kernel::of::<kernels::project_rows::Entry>(),
        projection(&[projected])?,
    )?;
    // A dense table converts by import, any other by repacking
    // (`CheckedPerLayerEntryEntries`).
    let conversion = if entry.table_source.dtype().is_some() && entry.table.dtype().is_some() {
        Kernel::of::<kernels::import_dense::Entry>()
    } else {
        Kernel::of::<kernels::repack_weight::Entry>()
    };
    step.call(
        conversion,
        Work {
            row_bytes: row_bytes(entry.table_source, channels)?,
            upload_row_bytes: table.bytes / table_rows,
            ..Work::default()
        },
    )?;
    planned(
        load,
        WeightScope::Target,
        WeightKind::PerLayerProjectionNorm,
        entry.norm,
    )?;
    let f32_rows = row_bytes(Element::f32(), channels)?;
    step.call(
        Kernel::of::<kernels::per_layer_inputs::Entry>(),
        rows(f32_rows),
    )?;
    step.call(
        Kernel::of::<kernels::conditioning_overlay::Entry>(),
        rows(f32_rows),
    )
}

/// `operators::streams::graph::mix_site` in `scope`: `stream_mix_down`
/// streams the site's norm and down weights (and, `injected`, its inject
/// weights), and `stream_mix_up` its up weights and norm. The consumer entry
/// sums the partials in its prologue, and the sublayer's output entry
/// scatters into the streams in its epilogue: neither is a call.
fn mix_site(
    step: &mut Step<'_>,
    load: &ModelLoadPlan,
    scope: WeightScope,
    site: MixSite,
    inject: Option<Element>,
) -> Result<(), AssessmentError> {
    let weight = |kind, bound| planned(load, scope, kind, bound);
    let norm = weight(WeightKind::HyperConnectionNorm, site.norm)?;
    let mut down = vec![norm, weight(WeightKind::HyperConnectionDown, site.down)?];
    if let Some(inject) = inject {
        down.push(weight(WeightKind::HyperConnectionInject, inject)?);
    }
    step.call(
        Kernel::of::<kernels::stream_mix_down::Entry>(),
        projection(&down)?,
    )?;
    step.call(
        Kernel::of::<kernels::stream_mix_up::Entry>(),
        projection(&[weight(WeightKind::HyperConnectionUp, site.up)?, norm])?,
    )
}

/// The input of the sublayer at `scope` as it runs: the residual under its
/// RMS input norm, which the consumer entry fuses (bound at `norm`), or the
/// mix site of a hyper-connection sublayer.
fn sublayer_input(
    step: &mut Step<'_>,
    load: &ModelLoadPlan,
    definition: &ModelDefinition,
    scope: WeightScope,
    norm: Element,
    streams: Option<StreamsBinding>,
) -> Result<(), AssessmentError> {
    let WeightScope::TargetSublayer(index) = scope else {
        return Err(demand_error(format!("{scope:?} is not a target sublayer")));
    };
    let sublayer = definition
        .decoder
        .blocks
        .get(index.block as usize)
        .and_then(|block| block.sublayers.get(index.sublayer as usize))
        .ok_or_else(|| demand_error(format!("{scope:?} is not a decoder sublayer")))?;
    match &sublayer.input {
        InputNorm::HyperConnectionMix(_) => {
            let site = streams
                .and_then(|streams| streams.sublayer(index.sublayer))
                .ok_or_else(|| demand_error(format!("{scope:?} mixes without its site")))?;
            mix_site(step, load, scope, site.site, Some(site.inject))
        }
        _ => planned(load, scope, WeightKind::InputNorm, norm).map(|_| ()),
    }
}

/// `operators::ngram` in `scope`, whose bank components (its window, then
/// its token window) start at `component`: the row rule over the token
/// window (`ngram_rows`), the read of its heads' stored rows from the
/// table's slots (`stored_rows`), the key and value projections
/// (`project_rows`), the gate over the key and query norms (`ngram_gate`),
/// and the row op over the convolution norm, the taps and the window
/// (`ngram_inject`).
fn ngram(
    step: &mut Step<'_>,
    load: &ModelLoadPlan,
    layout: &ModelStateLayout,
    scope: WeightScope,
    binding: crate::NGramBinding,
    component: usize,
    index: usize,
) -> Result<(), AssessmentError> {
    let weight = |kind, bound| planned(load, scope, kind, bound);
    step.call(
        Kernel::of::<kernels::ngram_rows::Entry>(),
        sequences(bank_bytes(layout, component + 1, 1, index)?),
    )?;
    let table = weight(WeightKind::NGramTable, binding.table)?;
    let slots = table.shape[0];
    if slots == 0 || table.resident_bytes % slots != 0 {
        return Err(demand_error(format!(
            "n-gram table {:?} does not split into its slots",
            table.descriptor.name
        )));
    }
    step.call(
        Kernel::of::<kernels::stored_rows::Entry>(),
        rows(
            (table.resident_bytes / slots)
                .checked_mul(binding.heads())
                .ok_or_else(|| demand_error("n-gram row bytes overflow"))?,
        ),
    )?;
    let project = Kernel::of::<kernels::project_rows::Entry>();
    step.call(
        project,
        projection(&[weight(WeightKind::NGramKey, binding.key)?])?,
    )?;
    step.call(
        project,
        projection(&[weight(WeightKind::NGramValue, binding.value)?])?,
    )?;
    step.call(
        Kernel::of::<kernels::ngram_gate::Entry>(),
        weights(&[
            weight(WeightKind::NGramKeyNorm, binding.norm)?,
            weight(WeightKind::NGramQueryNorm, binding.norm)?,
        ])?,
    )?;
    step.call(
        Kernel::of::<kernels::ngram_inject::Entry>(),
        Work {
            weights: weight_bytes(&[
                weight(WeightKind::NGramConvolutionNorm, binding.norm)?,
                weight(WeightKind::NGramConvolution, Element::f32())?,
            ])?,
            sequence_bytes: bank_bytes(layout, component, 1, index)?,
            ..Work::default()
        },
    )
}

/// `operators::routed::graph` at decode rows in `scope`: the shared
/// expert's expansion (`dense_expand`, or `dense_up` up-only) and down
/// projection onto the sum's root (`dense_output`), selection
/// (`routed_select`), the latent projections (`project_rows`,
/// `dense_output`), and the selected experts' expansion (`routed_gate_up`,
/// or `routed_up` up-only) and down projection (`routed_down`). Its input
/// is the caller's.
fn general_routed(
    step: &mut Step<'_>,
    load: &ModelLoadPlan,
    scope: WeightScope,
    binding: &GeneralRoutedBinding,
) -> Result<(), AssessmentError> {
    let weight = |kind, bound| planned(load, scope, kind, bound);
    let shape = binding.shape;
    if let Some((gate, up, down)) = binding.shared {
        let up = weight(WeightKind::SharedUp, up)?;
        match gate {
            Some(gate) => step.call(
                Kernel::of::<kernels::dense_expand::Entry>(),
                projection(&[weight(WeightKind::SharedGate, gate)?, up])?,
            )?,
            None => step.call(Kernel::of::<kernels::dense_up::Entry>(), projection(&[up])?)?,
        }
        step.call(
            Kernel::of::<kernels::dense_output::Entry>(),
            projection(&[weight(WeightKind::SharedDown, down)?])?,
        )?;
    }
    step.call(
        Kernel::of::<kernels::routed_select::Entry>(),
        projection(&[weight(WeightKind::Router, binding.router)?])?,
    )?;
    if let Some((down, up)) = binding.latent {
        step.call(
            Kernel::of::<kernels::project_rows::Entry>(),
            projection(&[weight(WeightKind::LatentDown, down)?])?,
        )?;
        step.call(
            Kernel::of::<kernels::dense_output::Entry>(),
            projection(&[weight(WeightKind::LatentUp, up)?])?,
        )?;
    }
    let up = weight(WeightKind::ExpertUp, binding.expert_up)?;
    let (expansion, tensors) = match binding.expert_gate {
        Some(gate) => (
            Kernel::of::<kernels::routed_gate_up::Entry>(),
            vec![weight(WeightKind::ExpertGate, gate)?, up],
        ),
        None => (Kernel::of::<kernels::routed_up::Entry>(), vec![up]),
    };
    let chosen = |tensors: &[&WeightPlan]| experts(tensors, shape.experts, shape.selected);
    step.call(
        expansion,
        Work {
            experts: Some(chosen(&tensors)?),
            ..Work::default()
        },
    )?;
    step.call(
        Kernel::of::<kernels::routed_down::Entry>(),
        Work {
            experts: Some(chosen(&[weight(
                WeightKind::ExpertDown,
                binding.expert_down,
            )?])?),
            ..Work::default()
        },
    )
}

fn usize_bytes(bytes: usize) -> Result<u64, AssessmentError> {
    u64::try_from(bytes).map_err(|_| demand_error("state bytes exceed u64"))
}

/// What one decode row of `layer`'s attention reads: its history's bytes per
/// token and, for a window domain, the most tokens it reads. A layer sharing
/// another layer's history reads the source's.
fn history_read(
    domains: &[HistoryDomainLayout],
    layer: LayerRef,
) -> Result<(u64, Option<u64>), AssessmentError> {
    for domain in domains {
        let window = match domain {
            HistoryDomainLayout::Token { .. } => None,
            // A window domain's rows are its window's tokens.
            HistoryDomainLayout::Window { rows, .. } => Some(usize_bytes(*rows)?),
            // An indexer's block keys: no attention layer's history.
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

/// One decode row's `indexer_select` call for target block `layer`'s
/// selected read of `shape`: it reads the raw keys of the positions before
/// it in its block and writes its own and the block it completes, then
/// scores every complete block (reading its key, writing and reading its F32
/// score) and selects. Per token of context that is one block's key and
/// score traffic over the block's rows; its scoring arithmetic is priced by
/// those bytes. Its launch clauses read its bitmap words (`B`) and, at
/// decode rows, its selected segments (`C`, `operators::attention::graph::
/// select`).
fn indexer_select(
    step: &mut Step<'_>,
    domains: &[HistoryDomainLayout],
    layer: u32,
    shape: crate::SelectionShape,
) -> Result<(), AssessmentError> {
    let indexer = LayerRef::Indexer(layer);
    let mut raw = None;
    let mut finished = None;
    for domain in domains {
        match domain {
            HistoryDomainLayout::Window { .. } if raw.is_none() => {
                raw = component_row_bytes(domain, indexer)?;
            }
            HistoryDomainLayout::Block { .. } if finished.is_none() => {
                finished = component_row_bytes(domain, indexer)?;
            }
            _ => {}
        }
    }
    let (Some(raw), Some(finished)) = (raw, finished) else {
        return Err(demand_error(format!(
            "selected attention layer {layer} has no indexer history"
        )));
    };
    let block = shape.block;
    let local = raw
        .checked_mul(block)
        .and_then(|bytes| bytes.checked_add(finished))
        .ok_or_else(|| demand_error("indexer row bytes overflow"))?;
    let scored = finished
        .checked_add(2 * 4)
        .ok_or_else(|| demand_error("indexer block bytes overflow"))?;
    step.call_with(
        Kernel::of::<kernels::indexer_select::Entry>(),
        &[("B", shape.words), ("C", selected_segments(shape.blocks))],
        Work {
            row_bytes: local,
            history: Some(HistoryRead {
                layout: None,
                bytes_per_token: scored.div_ceil(block),
                flops_per_token: 0,
                window: None,
            }),
            ..Work::default()
        },
    )
}

/// The bytes of `count` recurrent bank components from `first`.
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

impl DecodeDemand {
    /// The calls of one plain decode step, from the model's program plan,
    /// resident load plan and state layout, each naming the entry its graph
    /// runs at decode rows on the declarations' backend.
    pub fn from_model(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
        declarations: &Declarations,
    ) -> Result<Self, AssessmentError> {
        let program = load
            .program_plan(definition, codec)
            .map_err(|error| demand_error(error.to_string()))?;
        let layout =
            ModelStateLayout::derive(&definition.decoder, &[], codec, 0).map_err(demand_error)?;
        let backend = declarations.backend();
        let target = program.target();
        let mut step = Step {
            declarations,
            resolved: HashMap::new(),
            calls: Vec::new(),
        };

        let embedding = target.embedding();
        let table = planned(
            load,
            WeightScope::Target,
            WeightKind::Embedding,
            embedding.table,
        )?;
        step.call(
            Kernel::of::<kernels::embedding_rows::Entry>(),
            rows(table.resident_bytes / table.shape[0]),
        )?;
        if let Some(entry) = target.per_layer() {
            per_layer_entry(&mut step, load, entry)?;
        }

        // The first bank component of the next recurrent mixer.
        let mut recurrent_component = 0usize;
        for (index, block) in target.blocks().iter().enumerate() {
            let layer =
                u32::try_from(index).map_err(|_| demand_error("block index exceeds u32"))?;
            let [mixer_scope, feed_forward_scope] =
                crate::programs::native_target_graph::block_scopes(index).map_err(demand_error)?;
            let streams = block.streams();
            // Block 0 widens the embedded row into the streams
            // (`repeat_rows`), which streams no weight.
            if index == 0 && streams.is_some() {
                step.call(Kernel::of::<kernels::repeat_rows::Entry>(), Work::default())?;
            }
            let input = |step: &mut Step<'_>, scope, norm| {
                sublayer_input(step, load, definition, scope, norm, streams)
            };
            let scope = mixer_scope;
            let weight = |kind, bound| planned(load, scope, kind, bound);
            let output = Kernel::of::<kernels::attention_output::Entry>();
            match block.mixer() {
                MixerProgramSlot::Attention(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    let shape = binding.shape;
                    let query = if shape.interleaved_gate > 0 {
                        WeightKind::QueryGate
                    } else {
                        WeightKind::Query
                    };
                    let projections = [
                        (true, query, binding.query),
                        (
                            shape.gate_rows() > 0,
                            WeightKind::AttentionGate,
                            binding.gate,
                        ),
                        (shape.key_rows() > 0, WeightKind::Key, binding.key),
                        (shape.value_rows() > 0, WeightKind::Value, binding.value),
                    ]
                    .into_iter()
                    .filter(|(present, _, _)| *present)
                    .map(|(_, kind, bound)| weight(kind, bound))
                    .collect::<Result<Vec<_>, _>>()?;
                    let project = Kernel::of::<kernels::attention_project::Entry>();
                    step.call(project, projection(&projections)?)?;
                    let attend = match binding.history.encoding() {
                        HistoryEncoding::Dense => Kernel::of::<kernels::attention_decode::Entry>(),
                        HistoryEncoding::Affine => {
                            Kernel::of::<kernels::attention_decode_k8v4::Entry>()
                        }
                        HistoryEncoding::Unsupported => {
                            return Err(demand_error(
                                "rotated K4/V4 history has no native attention entry",
                            ));
                        }
                    };
                    let (row_bytes, window) =
                        history_read(&layout.target_history, LayerRef::Target(layer))?;
                    let flops_per_token = 2 * shape.kv_heads * shape.group * (2 * shape.width);
                    let read = |window| HistoryRead {
                        layout: Some(codec),
                        bytes_per_token: row_bytes,
                        flops_per_token,
                        window,
                    };
                    match binding.selection {
                        None => step.call(
                            attend,
                            Work {
                                history: Some(read(window)),
                                ..Work::default()
                            },
                        )?,
                        // The indexer projects its query and key, selects,
                        // and attention reads the selected blocks and the
                        // row's tail: at most one block past them.
                        Some(selection) => {
                            step.call(
                                project,
                                projection(&[
                                    weight(WeightKind::IndexerQuery, selection.query)?,
                                    weight(WeightKind::IndexerKey, selection.key)?,
                                ])?,
                            )?;
                            indexer_select(
                                &mut step,
                                &layout.target_history,
                                layer,
                                selection.shape,
                            )?;
                            let selected = (selection.shape.blocks + 1)
                                .checked_mul(selection.shape.block)
                                .ok_or_else(|| demand_error("selected tokens overflow"))?;
                            step.call(
                                attend,
                                Work {
                                    history: Some(read(Some(
                                        window.map_or(selected, |window| window.min(selected)),
                                    ))),
                                    ..Work::default()
                                },
                            )?;
                        }
                    }
                    let projected = weight(WeightKind::AttentionOutput, binding.output)?;
                    tail(&mut step, binding.tail, output, projected, weight)?;
                }
                // The query-only projection row (`attention_project`), the
                // state advance over the layer's bank (window, state and
                // tape components, `state_space_step`), the gated group norm
                // (`state_space_gate`) and the output projection
                // (`operators::state_space`).
                MixerProgramSlot::StateSpace(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    step.call(
                        Kernel::of::<kernels::attention_project::Entry>(),
                        projection(&[weight(
                            WeightKind::StateSpaceProjection,
                            binding.projection,
                        )?])?,
                    )?;
                    step.call(
                        Kernel::of::<kernels::state_space_step::Entry>(),
                        sequences(bank_bytes(&layout, recurrent_component, 3, index)?),
                    )?;
                    recurrent_component += 3;
                    step.call(
                        Kernel::of::<kernels::state_space_gate::Entry>(),
                        weights(&[weight(WeightKind::StateSpaceNorm, Element::f32())?])?,
                    )?;
                    step.call(
                        output,
                        projection(&[weight(WeightKind::RecurrentOutput, binding.output)?])?,
                    )?;
                }
                // The segmented `u | C` projection (`short_conv_project`),
                // the gated taps over the layer's window (its only bank
                // component, `short_conv_rows`) and the output projection
                // (`operators::short_conv`).
                MixerProgramSlot::ShortConv(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    step.call(
                        Kernel::of::<kernels::short_conv_project::Entry>(),
                        projection(&[
                            weight(WeightKind::ShortConvInputGate, binding.input_gate)?,
                            weight(WeightKind::ShortConvOutputGate, binding.output_gate)?,
                            weight(WeightKind::ShortConvValue, binding.value)?,
                        ])?,
                    )?;
                    let taps = weight(WeightKind::RecurrentConvolution, Element::f32())?;
                    step.call(
                        Kernel::of::<kernels::short_conv_rows::Entry>(),
                        Work {
                            weights: weight_bytes(&[taps])?,
                            sequence_bytes: bank_bytes(&layout, recurrent_component, 1, index)?,
                            ..Work::default()
                        },
                    )?;
                    recurrent_component += 1;
                    let projected = weight(WeightKind::RecurrentOutput, binding.output)?;
                    tail(&mut step, binding.tail, output, projected, weight)?;
                }
                // The segmented projection, the step over the layer's
                // recurrent bank (window, delta and tape components), in the
                // backend's step form (`operators::gated_delta`), and the
                // output projection.
                MixerProgramSlot::Recurrent(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    let (project, advance) = match StepForm::of(backend).map_err(demand_error)? {
                        StepForm::Step => (
                            Kernel::of::<kernels::gated_delta_project::Entry>(),
                            Kernel::of::<kernels::gated_delta_step::Entry>(),
                        ),
                        StepForm::Convolved => (
                            Kernel::of::<kernels::gated_delta_project_convolved::Entry>(),
                            Kernel::of::<kernels::gated_delta_step_convolved::Entry>(),
                        ),
                    };
                    step.call(
                        project,
                        projection(&[
                            weight(WeightKind::RecurrentQueryKeyValue, binding.qkv)?,
                            weight(WeightKind::RecurrentGate, binding.gate)?,
                            weight(WeightKind::RecurrentAlpha, binding.alpha)?,
                            weight(WeightKind::RecurrentBeta, binding.beta)?,
                        ])?,
                    )?;
                    step.call(
                        advance,
                        sequences(bank_bytes(&layout, recurrent_component, 3, index)?),
                    )?;
                    recurrent_component += 3;
                    weight(WeightKind::RecurrentNorm, binding.recurrent_norm)?;
                    step.call(
                        output,
                        projection(&[weight(WeightKind::RecurrentOutput, binding.output)?])?,
                    )?;
                }
            }
            let scope = feed_forward_scope;
            let weight = |kind, bound| planned(load, scope, kind, bound);
            let Some(feed_forward) = block.feed_forward() else {
                continue;
            };
            match feed_forward {
                FeedForwardProgramSlot::Dense(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    step.call(
                        Kernel::of::<kernels::dense_expand::Entry>(),
                        projection(&[
                            weight(WeightKind::DenseGate, binding.gate)?,
                            weight(WeightKind::DenseUp, binding.up)?,
                        ])?,
                    )?;
                    let down = weight(WeightKind::DenseDown, binding.down)?;
                    tail(
                        &mut step,
                        binding.tail,
                        Kernel::of::<kernels::dense_output::Entry>(),
                        down,
                        weight,
                    )?;
                }
                // `operators::routed::fused_graph` at decode rows, in the
                // backend's decode form: routing (`routed_route`), then the
                // choices' and the shared expert's gate/up
                // (`routed_expand`); or routing with the shared expert's
                // gate/up (`routed_route_shared`), then the choices'
                // (`routed_gate_up`). Then the choices' and the shared
                // expert's down projections (`routed_output`).
                FeedForwardProgramSlot::Routed(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    let router = projection(&[weight(WeightKind::Router, binding.router)?])?;
                    let shared = projection(&[
                        weight(WeightKind::SharedGate, binding.shared_gate)?,
                        weight(WeightKind::SharedUp, binding.shared_up)?,
                    ])?;
                    let choices = Work {
                        experts: Some(experts(
                            &[
                                weight(WeightKind::ExpertGate, binding.expert_gate)?,
                                weight(WeightKind::ExpertUp, binding.expert_up)?,
                            ],
                            binding.experts,
                            binding.selected,
                        )?),
                        ..Work::default()
                    };
                    match DecodeForm::of(backend).map_err(demand_error)? {
                        DecodeForm::Expand => {
                            step.call(Kernel::of::<kernels::routed_route::Entry>(), router)?;
                            step.call(
                                Kernel::of::<kernels::routed_expand::Entry>(),
                                join([choices, shared])?,
                            )?;
                        }
                        DecodeForm::SharedRoute => {
                            step.call(
                                Kernel::of::<kernels::routed_route_shared::Entry>(),
                                join([router, shared])?,
                            )?;
                            step.call(Kernel::of::<kernels::routed_gate_up::Entry>(), choices)?;
                        }
                    }
                    let down = Work {
                        experts: Some(experts(
                            &[weight(WeightKind::ExpertDown, binding.expert_down)?],
                            binding.experts,
                            binding.selected,
                        )?),
                        ..Work::default()
                    };
                    step.call(
                        Kernel::of::<kernels::routed_output::Entry>(),
                        join([
                            down,
                            projection(&[weight(WeightKind::SharedDown, binding.shared_down)?])?,
                        ])?,
                    )?;
                }
                // `programs::graph::parallel`: the dense branch's expansion
                // (`dense_expand`) and its projection into F32 rows
                // (`project_rows`), the routed branch (summed onto zeros) in
                // its own scope, then `moe_tail`.
                FeedForwardProgramSlot::Parallel(binding) => {
                    let WeightScope::TargetSublayer(sublayer) = scope else {
                        return Err(demand_error("parallel branches outside a target sublayer"));
                    };
                    let [dense_scope, routed_scope] = DenseBesideRouted::scopes(sublayer);
                    let dense = binding.dense;
                    let weight = |kind, bound| planned(load, dense_scope, kind, bound);
                    weight(WeightKind::InputNorm, dense.norm)?;
                    step.call(
                        Kernel::of::<kernels::dense_expand::Entry>(),
                        projection(&[
                            weight(WeightKind::DenseGate, dense.gate)?,
                            weight(WeightKind::DenseUp, dense.up)?,
                        ])?,
                    )?;
                    step.call(
                        Kernel::of::<kernels::project_rows::Entry>(),
                        projection(&[weight(WeightKind::DenseDown, dense.down)?])?,
                    )?;
                    planned(
                        load,
                        routed_scope,
                        WeightKind::InputNorm,
                        binding.routed.norm,
                    )?;
                    general_routed(&mut step, load, routed_scope, &binding.routed)?;
                    // The branch and tail norms share the tail's element.
                    let norms = [dense_scope, routed_scope, scope]
                        .into_iter()
                        .map(|norm_scope| {
                            planned(load, norm_scope, WeightKind::PostNorm, binding.norm)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    step.call(Kernel::of::<kernels::moe_tail::Entry>(), weights(&norms)?)?;
                }
                FeedForwardProgramSlot::GeneralRouted(binding) => {
                    input(&mut step, scope, binding.norm)?;
                    general_routed(&mut step, load, scope, &binding)?;
                }
            }
            // `programs::graph::per_layer`: the gate over the layer's slice
            // of the per-layer rows (`per_layer_gate`), then its post-norm
            // tail.
            if let Some(binding) = block.per_layer() {
                let scope = WeightScope::TargetSublayer(SublayerIndex {
                    block: layer,
                    sublayer: 2,
                });
                let SublayerTail::PostNorm { norm, .. } = binding.tail else {
                    return Err(demand_error(
                        "a per-layer input sublayer without a post-norm tail",
                    ));
                };
                step.call(
                    Kernel::of::<kernels::per_layer_gate::Entry>(),
                    projection(&[planned(
                        load,
                        scope,
                        WeightKind::PerLayerGate,
                        binding.gate,
                    )?])?,
                )?;
                post_norm(
                    &mut step,
                    planned(
                        load,
                        scope,
                        WeightKind::PerLayerProjection,
                        binding.projection,
                    )?,
                    planned(load, scope, WeightKind::PostNorm, norm)?,
                )?;
            }
            if let Some(binding) = block.ngram() {
                let scope = WeightScope::TargetSublayer(SublayerIndex {
                    block: layer,
                    sublayer: 2,
                });
                ngram(
                    &mut step,
                    load,
                    &layout,
                    scope,
                    binding,
                    recurrent_component,
                    index,
                )?;
                recurrent_component += 2;
            }
        }

        // The last block's graph ends with the decoder's exit site, which
        // injects into nothing.
        if let Some(exit) = target
            .blocks()
            .last()
            .and_then(|block| block.streams()?.exit)
        {
            mix_site(&mut step, load, WeightScope::Target, exit, None)?;
        }

        // The selection readout graph: the features over the final row under
        // the exit norm (or the exit site's partials, unnormalized,
        // `readout_features_rows`), the vocabulary projection (its physical
        // matrix, even when tied to the embedding, `readout_head_rows`), and
        // sampling of the one F32 logits row (`sample_rows`).
        let readout = target.readout();
        let norm = if InputForm::exit(&definition.decoder).normalized {
            vec![planned(
                load,
                WeightScope::Target,
                WeightKind::OutputNorm,
                readout.norm,
            )?]
        } else {
            Vec::new()
        };
        step.call(
            Kernel::of::<kernels::readout_features_rows::Entry>(),
            weights(&norm)?,
        )?;
        match readout.head {
            // Admission keeps the canonical full-projection demand even
            // when bounded classes have an optional qualified implementation.
            ReadoutHead::Packed { weight, .. } | ReadoutHead::Qualified { weight, .. } => {
                step.call(
                    Kernel::of::<kernels::readout_head_rows::Entry>(),
                    projection(&[planned(
                        load,
                        WeightScope::Target,
                        WeightKind::Output,
                        weight,
                    )?])?,
                )?;
            }
        }
        step.call(
            Kernel::of::<kernels::sample_rows::Entry>(),
            rows(
                definition
                    .decoder
                    .vocabulary
                    .checked_mul(4)
                    .ok_or_else(|| demand_error("logits bytes overflow"))?,
            ),
        )?;
        Ok(Self { calls: step.calls })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use seismic::{BackendName, Layout};

    /// The declarations of a load of `definition` under `codec` on
    /// `backend`, from its kernel inventory.
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
            crate::native::TuningLimits::of(crate::assessment::assess::tests::LIMITS, definition),
        )
    }

    /// The decode demand of `definition` under `codec` on `backend`.
    pub(crate) fn demand_on(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        codec: KvCodec,
        backend: BackendName,
    ) -> Result<DecodeDemand, AssessmentError> {
        let declarations = declarations(definition, load, codec, backend)?;
        DecodeDemand::from_model(definition, load, codec, &declarations)
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

    fn histories(demand: &DecodeDemand) -> Vec<HistoryRead> {
        demand
            .calls
            .iter()
            .filter_map(|call| call.work.history)
            .collect()
    }

    fn fixture() -> (ModelDefinition, ModelLoadPlan) {
        let definition = crate::planning::tests::fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows8,
        )
        .unwrap();
        (definition, load)
    }

    fn resident(load: &ModelLoadPlan, kind: WeightKind) -> &WeightPlan {
        load.target()
            .iter()
            .find(|weight| weight.role.kind == kind)
            .unwrap()
    }

    #[test]
    fn a_plain_decode_step_streams_every_planned_weight_once() {
        let (definition, load) = fixture();
        let demand = demand(&definition, &load, KvCodec::Dense).unwrap();
        // Embedding, attention projection, attention, attention output,
        // dense expansion, dense output, readout features, readout head,
        // sampling.
        assert_eq!(
            demand
                .calls
                .iter()
                .map(|call| call.launches.kernel().name())
                .collect::<Vec<_>>(),
            [
                "embedding_rows",
                "attention_project",
                "attention_decode",
                "attention_output",
                "dense_expand",
                "dense_output",
                "readout_features_rows",
                "readout_head_rows",
                "sample_rows",
            ]
        );
        let matrices = [
            WeightKind::QueryGate,
            WeightKind::Key,
            WeightKind::Value,
            WeightKind::AttentionOutput,
            WeightKind::DenseGate,
            WeightKind::DenseUp,
            WeightKind::DenseDown,
            WeightKind::Output,
        ];
        let weights = matrices
            .iter()
            .chain(&[WeightKind::OutputNorm])
            .map(|kind| resident(&load, *kind).resident_bytes)
            .sum::<u64>();
        let vocabulary = definition.decoder.vocabulary;
        let embedding = resident(&load, WeightKind::Embedding);
        let embedding_row = embedding.resident_bytes / embedding.shape[0];
        assert_eq!(streamed(&demand), weights + embedding_row + vocabulary * 4);
        // Every matrix's FLOPs are 2 × its elements per row.
        let flops = matrices
            .iter()
            .map(|kind| 2 * resident(&load, *kind).shape.iter().product::<u64>())
            .sum::<u64>();
        assert_eq!(
            demand
                .calls
                .iter()
                .flat_map(|call| &call.work.row_flops)
                .map(|flops| flops.flops)
                .sum::<u64>(),
            flops
        );
    }

    /// Over hyper-connection streams a sublayer reads through its mix site
    /// and the exit site precedes the readout, which reads no norm.
    #[test]
    fn a_streams_decode_step_streams_its_mix_sites_and_no_norms() {
        let definition = crate::planning::tests::hyper_connection_fixture_definition();
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows8,
        )
        .unwrap();
        let demand = demand(&definition, &load, KvCodec::Dense).unwrap();
        // Embedding, widening, mix site (two stages), attention projection,
        // attention, attention output, exit site (two stages), readout
        // features, readout head, sampling.
        assert_eq!(demand.calls.len(), 12);
        let mixer = WeightScope::TargetSublayer(SublayerIndex {
            block: 0,
            sublayer: 0,
        });
        let bytes = |scope, kind| {
            load.target()
                .iter()
                .find(|weight| weight.role.scope == scope && weight.role.kind == kind)
                .unwrap()
                .resident_bytes
        };
        // Each site streams its norm with both stages.
        let site = |scope, injected: bool| {
            2 * bytes(scope, WeightKind::HyperConnectionNorm)
                + bytes(scope, WeightKind::HyperConnectionDown)
                + bytes(scope, WeightKind::HyperConnectionUp)
                + if injected {
                    bytes(scope, WeightKind::HyperConnectionInject)
                } else {
                    0
                }
        };
        let weights = [
            WeightKind::QueryGate,
            WeightKind::Key,
            WeightKind::Value,
            WeightKind::AttentionOutput,
        ]
        .iter()
        .map(|kind| bytes(mixer, *kind))
        .sum::<u64>()
            + bytes(WeightScope::Target, WeightKind::Output);
        let embedding = resident(&load, WeightKind::Embedding);
        let embedding_row = embedding.resident_bytes / embedding.shape[0];
        assert_eq!(
            streamed(&demand),
            weights
                + site(mixer, true)
                + site(WeightScope::Target, false)
                + embedding_row
                + definition.decoder.vocabulary * 4
        );
    }

    /// An n-gram input charges its row rule over its token window, its
    /// heads' stored rows, its key and value projections, its gate's norms,
    /// and its row op's norm, taps and window; the recurrent mixers after it
    /// read their own bank components.
    #[test]
    fn an_ngram_input_streams_its_rows_projections_and_windows() {
        let load_of = |definition: &ModelDefinition| {
            let manifest = crate::planning::tests::stored_table_manifest(definition);
            ModelLoadPlan::derive(
                &manifest,
                definition,
                crate::ComponentSelection {
                    head: false,
                    vision: false,
                },
                Layout::Rows8,
            )
            .unwrap()
        };
        let definition = crate::planning::tests::qwen4exp_fixture_definition();
        let mut plain = definition.clone();
        plain.decoder.blocks[0].sublayers.truncate(2);
        let load = load_of(&definition);
        let with = demand(&definition, &load, KvCodec::Dense).unwrap();
        let without = demand(&plain, &load_of(&plain), KvCodec::Dense).unwrap();
        // Embedding and widening; per block a mix site (two stages) before
        // its mixer and its feed-forward; the gated delta's projection, step
        // and output; the routed router, expansion and output; the
        // attention's projection, read and output; the exit site and the
        // readout's three.
        assert_eq!(without.calls.len(), 2 + 2 * 2 * 2 + 3 + 3 + 3 + 3 + 2 + 3);
        assert_eq!(with.calls.len(), without.calls.len() + 6);
        let scope = WeightScope::TargetSublayer(SublayerIndex {
            block: 0,
            sublayer: 2,
        });
        let bytes = |kind| {
            load.target()
                .iter()
                .find(|weight| weight.role.scope == scope && weight.role.kind == kind)
                .unwrap()
        };
        let table = bytes(WeightKind::NGramTable);
        let heads = 4;
        let layout = ModelStateLayout::derive(&definition.decoder, &[], KvCodec::Dense, 0).unwrap();
        // The n-gram's windows follow the gated delta's three components.
        let windows = layout.target_recurrent[3..5]
            .iter()
            .map(|component| component.bytes().unwrap() as u64)
            .sum::<u64>();
        let ngram = heads * table.resident_bytes / table.shape[0]
            + [
                WeightKind::NGramKey,
                WeightKind::NGramValue,
                WeightKind::NGramKeyNorm,
                WeightKind::NGramQueryNorm,
                WeightKind::NGramConvolutionNorm,
                WeightKind::NGramConvolution,
            ]
            .iter()
            .map(|kind| bytes(*kind).resident_bytes)
            .sum::<u64>()
            + windows;
        assert_eq!(streamed(&with), streamed(&without) + ngram);
    }

    #[test]
    fn history_bytes_per_token_follow_the_state_layout_and_codec() {
        let (definition, load) = fixture();
        // One kv head of width 64, a group of two: dense bf16 keys and
        // values, 2 × 64 × 2 bytes; each row scores and accumulates both
        // heads of the group over 64 columns each.
        let dense = demand(&definition, &load, KvCodec::Dense).unwrap();
        let flops_per_token = 2 * 2 * (64 + 64);
        assert_eq!(
            histories(&dense),
            [HistoryRead {
                layout: Some(KvCodec::Dense),
                bytes_per_token: 256,
                flops_per_token,
                window: None,
            }]
        );
        // Affine: 64 key bytes, 32 value bytes and two (scale, zero) f16
        // pairs per 32-value group for each.
        let affine = demand(&definition, &load, KvCodec::AffineK8V4).unwrap();
        assert_eq!(
            histories(&affine),
            [HistoryRead {
                layout: Some(KvCodec::AffineK8V4),
                bytes_per_token: 64 + 32 + 2 * (2 * 2 * 2),
                flops_per_token,
                window: None,
            }]
        );
        assert!(matches!(
            demand(&definition, &load, KvCodec::RotatedK4V4),
            Err(AssessmentError::Demand(_))
        ));
    }

    /// A selected read charges its indexer: the projection of its query and
    /// key, a select call writing its raw key and block and reading its
    /// block's raw keys, and per token one block's F32 key and score traffic
    /// over the block's rows; attention reads the selected blocks and the
    /// tail alone.
    #[test]
    fn a_selected_read_charges_its_indexer_and_reads_its_blocks() {
        use magnitude_family_contracts::{
            HistoryReads, Indexer, Operator, RmsNorm, WeightDescriptor,
        };
        let (mut definition, _) = fixture();
        let hidden = definition.decoder.hidden;
        let Operator::Attention(attention) = &mut definition.decoder.blocks[0].sublayers[0].op
        else {
            panic!("the fixture's mixer is attention");
        };
        let norm = |name: &str| RmsNorm {
            weight: WeightDescriptor::stored(name, vec![64]),
            epsilon: 1e-6,
        };
        attention.reads = HistoryReads::Selected {
            block: 4,
            blocks: 8,
            indexer: Indexer {
                heads: 2,
                width: 64,
                query: WeightDescriptor::stored("indexer_q", vec![128, hidden]),
                key: WeightDescriptor::stored("indexer_k", vec![64, hidden]),
                query_norm: norm("indexer_q_norm"),
                key_norm: norm("indexer_k_norm"),
            },
        };
        let manifest = crate::planning::tests::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows8,
        )
        .unwrap();
        let (plain_definition, plain_load) = fixture();
        let plain = demand(&plain_definition, &plain_load, KvCodec::Dense).unwrap();
        let selected = demand(&definition, &load, KvCodec::Dense).unwrap();
        // The indexer's projection and select calls.
        assert_eq!(selected.calls.len(), plain.calls.len() + 2);
        // Its projection weights; the select call's raw keys (four F32 rows
        // of 64 plus their coordinates) and finished block key.
        let projection = resident(&load, WeightKind::IndexerQuery).resident_bytes
            + resident(&load, WeightKind::IndexerKey).resident_bytes;
        let raw = 64 * 4 + 4 * 4;
        assert_eq!(
            streamed(&selected),
            streamed(&plain) + projection + 4 * raw + 64 * 4
        );
        // The indexer reads one block key and two scores per four tokens;
        // attention at most the eight selected blocks and its tail.
        let history = histories(&selected);
        assert_eq!(history[0].bytes_per_token, (64 * 4 + 8) / 4);
        assert_eq!(history[0].window, None);
        assert_eq!(history[1].bytes_per_token, 256);
        assert_eq!(history[1].window, Some(9 * 4));
    }

    #[test]
    fn a_window_reads_at_most_its_tokens() {
        let read = HistoryRead {
            layout: Some(KvCodec::Dense),
            bytes_per_token: 10,
            flops_per_token: 0,
            window: Some(100),
        };
        assert_eq!(read.tokens_at(40), 40);
        assert_eq!(read.tokens_at(4_000), 100);
    }

    #[test]
    fn a_selected_expert_share_reads_its_slabs() {
        let (_, load) = fixture();
        let mut tensor = resident(&load, WeightKind::DenseUp).clone();
        tensor.shape = vec![8, 32, 64];
        tensor.resident_bytes = 8 * 32 * 64 * 2;
        assert_eq!(
            experts(&[&tensor], 8, 2).unwrap(),
            Experts {
                experts: 8,
                selected: 2,
                slab_bytes: vec![Bytes {
                    format: tensor.resident,
                    bytes: 32 * 64 * 2,
                }],
                slab_flops: vec![Flops {
                    format: tensor.resident,
                    flops: 2 * 32 * 64,
                }],
            }
        );
        assert!(experts(&[&tensor], 3, 1).is_err());
        assert!(experts(&[&tensor], 8, 9).is_err());
    }
}
