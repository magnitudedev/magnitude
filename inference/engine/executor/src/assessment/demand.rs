//! Header-derived decode demand: one record per call of a plain target
//! decode step, in the order the step runs them. Each names the entry it
//! runs ([`Launches`], whose launches at a row count are the backend's
//! declaration's) and the work it does: the weights it reads once per step,
//! the bytes and FLOPs of each row, the state of each sequence, and the
//! history it reads per token of context. Derived from the same program and
//! load plans native preparation uses, and from the state layout the memory
//! terms use for state and history bytes.
//!
//! A plan that banks its experts decodes its streamed layers in their
//! streamed form (`operators::routed::streamed`) while its bank lacks any
//! expert, which its demand is: each such layer's route resolves and hints
//! (`routed_route_streamed`, `routed_select_streamed`), the first sets the
//! run's step word (`routed_step`), each lands its absent choices
//! (`routed_land`), and a layer with a CPU site requests
//! the CPU's choices (`routed_site_request`) and combines with its answer
//! (`routed_output_project` and `routed_output_sited`, or the general
//! form's down), all at any misses, none included. Where the backend's
//! parks are submission boundaries (`BackendName::parks_in_kernels`), each
//! landing launch's park, site request's publication and sited combine's park
//! ends one ([`DecodeDemand::boundaries`]). A plan that holds its experts
//! whole decodes its resident forms alone.
//!
//! Matmul FLOPs are 2 × the weight's logical elements per row (a multiply
//! and an add per weight), of a routed tensor the selected experts' slabs.
//! Attention FLOPs per row and token read are 2 · KV · G · (W_k + W_v): each
//! query head scores a key and accumulates a value of its head width.
//! Recurrent and state-space state advances are priced by their bytes only:
//! their arithmetic per state element is the kernels', which no plan
//! states.

use super::launches::{operand_recipe, Declarations, Kernel, Launches, Words};
use super::AssessmentError;
use crate::operators::attention::graph::selected_segments;
use crate::operators::attention::pooled::Pooling;
use crate::operators::gated_delta::graph::StepForm;
use crate::operators::routed::{fused_graph::DecodeForm, streamed};
use crate::operators::streams::{BlockStreams, MixSite, SinkhornBinding};
use crate::operators::{parallel::DenseBesideRouted, InputForm};
use crate::{
    CandidatesBinding, FeedForwardProgramSlot, GeneralRoutedBinding, MixerProgramSlot,
    ModelLoadPlan, NGramGateForm, PerLayerEntryBinding, PickedBinding, ReadoutHead,
    SublayerTail, WeightPlan,
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
    /// The submissions the step's calls end beyond its own: its streamed
    /// layers' parks and publications where the backend's parks are
    /// submission boundaries, each a submit and a wait beside device time.
    pub boundaries: u64,
}

/// The words a banked plan's decode is encoded under, its streamed form: no
/// layer's group whole (`POLICY_WHOLE` clear) and none resolved
/// (`POLICY_RESOLVED` clear, the next layer's too), as while its bank lacks
/// experts its host tier does not hold settled. A plan holding its experts
/// whole binds no word.
const STREAMED: &Words = &[
    ("policy", magnitude_kernels::constants::POLICY_WHOLE, false),
    ("policy", magnitude_kernels::constants::POLICY_RESOLVED, false),
    ("next_policy", magnitude_kernels::constants::POLICY_RESOLVED, false),
];

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

/// One expert's bytes and FLOPs per row of an expert tensor `[count, N,
/// K]`: one `[N, K]` matrix of its resident form, whether the stack is
/// resident (`count` its experts) or banked (`count` its group's records,
/// of which it charges its slots).
fn slab(weight: &WeightPlan) -> Result<(Bytes, Flops), AssessmentError> {
    let malformed = || demand_error(format!("expert tensor {:?} is no stack of matrices", weight.descriptor.name));
    let [_, n, k] = weight.shape[..] else {
        return Err(malformed());
    };
    let bytes = weight.resident.canonical_byte_len(&[1, n, k]).map_err(|_| malformed())?;
    Ok((
        Bytes {
            format: weight.resident,
            bytes,
        },
        Flops {
            format: weight.resident,
            flops: 2 * n * k,
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
        .map(|tensor| slab(tensor))
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

/// A call's resolution key: its entry, operand recipe, specialization
/// statics and stated dimensions.
type Resolution = (&'static str, OperandRecipe, Vec<(&'static str, u64)>, Vec<(&'static str, u64)>);

/// The step as it accumulates.
struct Step<'a> {
    declarations: &'a Declarations,
    /// The words its calls are encoded under ([`STREAMED`] of a banked plan).
    words: &'static Words,
    /// Each entry's launches at each operand recipe, specialization statics
    /// and stated dimensions, resolved once.
    resolved: HashMap<Resolution, Launches>,
    calls: Vec<Call>,
    boundaries: u64,
}

/// The operand recipe of the weights `work` reads, whole or as experts.
fn work_recipe(work: &Work) -> OperandRecipe {
    let experts = work.experts.iter().flat_map(|experts| &experts.slab_bytes);
    operand_recipe(work.weights.iter().chain(experts).map(|bytes| bytes.format))
}

impl Step<'_> {
    /// One call of `kernel` doing `work`.
    fn call(&mut self, kernel: Kernel, work: Work) -> Result<(), AssessmentError> {
        self.call_with(kernel, &[], work)
    }

    /// One call of `kernel` stating `dimensions` its launch clauses read.
    fn call_with(
        &mut self,
        kernel: Kernel,
        dimensions: &[(&'static str, u64)],
        work: Work,
    ) -> Result<(), AssessmentError> {
        self.call_specialized(kernel, &[], dimensions, work)
    }

    /// One call of `kernel`'s specializations with `statics` (the call's
    /// binding, where the load prepares the entry for several), stating
    /// `dimensions` its launch clauses read.
    fn call_specialized(
        &mut self,
        kernel: Kernel,
        statics: &[(&'static str, u64)],
        dimensions: &[(&'static str, u64)],
        work: Work,
    ) -> Result<(), AssessmentError> {
        let recipe = work_recipe(&work);
        self.call_as(kernel, recipe, statics, dimensions, work)
    }

    /// [`Self::call_specialized`] at operand recipe `recipe`, for a call
    /// whose operands are not its work's weights (a landing launch's are the
    /// bank's records).
    fn call_as(
        &mut self,
        kernel: Kernel,
        recipe: OperandRecipe,
        statics: &[(&'static str, u64)],
        dimensions: &[(&'static str, u64)],
        work: Work,
    ) -> Result<(), AssessmentError> {
        let key = (kernel.name(), recipe, statics.to_vec(), dimensions.to_vec());
        let launches = match self.resolved.get(&key) {
            Some(launches) => launches.clone(),
            None => {
                let launches =
                    self.declarations
                        .launches(kernel, key.1, statics, dimensions, self.words)?;
                self.resolved.insert(key, launches.clone());
                launches
            }
        };
        self.calls.push(Call { launches, work });
        Ok(())
    }

    /// A park or publication of a streamed layer: a submission boundary
    /// where the backend's parks are, nothing where kernels wait in place.
    fn park(&mut self) {
        if !self.declarations.backend().parks_in_kernels() {
            self.boundaries += 1;
        }
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

/// The bytes of `elements` F32 values.
fn f32_bytes(elements: u64) -> Result<u64, AssessmentError> {
    row_bytes(Element::f32(), elements)
}

/// `operators::streams::graph::sinkhorn_site` in `scope`:
/// `stream_mix_sinkhorn` reads each row's `[S, H]` streams and projects
/// them through the site's F32 function weights (`(2+S)·S × S·H`, with its
/// base and scale), and writes the sublayer's input `[H]`, the site's pre
/// `[S]` and its join values (post | comb, `[S+S²]`). Beside the
/// projection, the first of its `iterations` Sinkhorn passes normalizes the
/// comb's columns and each later one its rows and then its columns
/// (`2·iterations − 1` normalizations, none at zero), a sum and a division
/// per element each.
fn sinkhorn_site(
    step: &mut Step<'_>,
    load: &ModelLoadPlan,
    scope: WeightScope,
    binding: SinkhornBinding,
) -> Result<(), AssessmentError> {
    let SinkhornBinding {
        hidden,
        streams,
        mixing,
    } = binding;
    let weight = |kind| planned(load, scope, kind, Element::f32());
    let mut work = projection(&[
        weight(WeightKind::HyperConnectionFunction)?,
        weight(WeightKind::HyperConnectionBase)?,
        weight(WeightKind::HyperConnectionScale)?,
    ])?;
    work.row_bytes = f32_bytes(streams * hidden + hidden + streams + streams + streams * streams)?;
    add(
        &mut work.row_flops,
        Flops {
            format: Element::f32(),
            flops: (2 * u64::from(mixing.iterations)).saturating_sub(1) * 2 * streams * streams,
        },
    )?;
    step.call(Kernel::of::<kernels::stream_mix_sinkhorn::Entry>(), work)
}

/// The input of the sublayer at `scope` as it runs: the residual under its
/// RMS input norm, which the consumer entry fuses (bound at `norm`); the
/// low-rank mix site of a hyper-connection sublayer; or its Sinkhorn site,
/// whose input row the consumer entry reads under that norm.
fn sublayer_input(
    step: &mut Step<'_>,
    load: &ModelLoadPlan,
    definition: &ModelDefinition,
    scope: WeightScope,
    norm: Element,
    streams: Option<BlockStreams>,
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
            let without = || demand_error(format!("{scope:?} mixes without its site"));
            match streams.ok_or_else(without)? {
                BlockStreams::LowRank(streams) => {
                    let site = streams.sublayer(index.sublayer).ok_or_else(without)?;
                    mix_site(step, load, scope, site.site, Some(site.inject))
                }
                BlockStreams::Sinkhorn(binding) => {
                    sinkhorn_site(step, load, scope, binding)?;
                    planned(load, scope, WeightKind::InputNorm, norm).map(|_| ())
                }
            }
        }
        _ => planned(load, scope, WeightKind::InputNorm, norm).map(|_| ()),
    }
}

/// `operators::ngram` in `scope`, whose bank components (under a
/// convolution its window, then its token window) start at `component`:
/// the row rule over the token window (`ngram_rows`), the read of its
/// heads' stored rows from the table's slots (`stored_rows`), the key and
/// value projections (`project_rows`), the gate over its weights, the key
/// and query norms or their stored product (`ngram_gate`), and the row op:
/// the gated value's sum (`ngram_add`), or under a convolution the op over
/// its norm, its taps and its window (`ngram_inject`).
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
    let token_window = component + binding.bank_components() - 1;
    step.call(
        Kernel::of::<kernels::ngram_rows::Entry>(),
        sequences(bank_bytes(layout, token_window, 1, index)?),
    )?;
    let table = weight(WeightKind::NGramTable, binding.table)?;
    let row = table
        .resident
        .canonical_byte_len(&[1, table.table_width()])
        .map_err(|error| demand_error(format!("n-gram table {:?}: {error}", table.descriptor.name)))?;
    step.call(
        Kernel::of::<kernels::stored_rows::Entry>(),
        rows(
            row.checked_mul(binding.heads())
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
    let gate = match binding.gate {
        NGramGateForm::Norms => vec![
            weight(WeightKind::NGramKeyNorm, binding.norm)?,
            weight(WeightKind::NGramQueryNorm, binding.norm)?,
        ],
        NGramGateForm::Product => vec![weight(WeightKind::NGramGate, binding.norm)?],
    };
    step.call(Kernel::of::<kernels::ngram_gate::Entry>(), weights(&gate)?)?;
    let row_bytes = ngram_output_row_bytes(&binding)?;
    match binding.convolution {
        None => step.call(Kernel::of::<kernels::ngram_add::Entry>(), rows(row_bytes)),
        Some(_) => step.call(
            Kernel::of::<kernels::ngram_inject::Entry>(),
            Work {
                weights: weight_bytes(&[
                    weight(WeightKind::NGramConvolutionNorm, binding.norm)?,
                    weight(WeightKind::NGramConvolution, Element::f32())?,
                ])?,
                row_bytes,
                sequence_bytes: bank_bytes(layout, component, 1, index)?,
                ..Work::default()
            },
        ),
    }
}

/// The device bytes each row of an n-gram input's output op (`ngram_add`,
/// or `ngram_inject` under a convolution) reads and writes: it reads the
/// row's `[S, H]` streams, its gates and its gated value `[H]`, and writes
/// its `[S, H]` streams. Its gates are the `S` gate values, and under a
/// convolution also the `S` inverse key norms the convolution's input is
/// formed with (`ngram_gate`'s `[2S]` scales).
fn ngram_output_row_bytes(binding: &crate::NGramBinding) -> Result<u64, AssessmentError> {
    let gates = binding.streams * (1 + u64::from(binding.convolution.is_some()));
    f32_bytes(2 * binding.streams * binding.hidden + gates + binding.hidden)
}

/// The decode entries' `XS` of the routed layer of `scope`, as its graph
/// states it (`streamed::source_rows`).
fn sources(load: &ModelLoadPlan, scope: WeightScope) -> [(&'static str, u64); 1] {
    [("XS", streamed::source_rows(streamed::served(load, scope).is_some()))]
}

/// The routing of the routed layer of `scope` doing `work`: `resident` of a
/// layer the load holds whole; of a streamed one its streamed form
/// (`operators::routed::streamed`): the run's step word where the layer
/// sets it (`routed_step`), its route `streamed` at its `AHEAD`, reading
/// beside `work` the hinted weights of each layer it predicts
/// (`ExpertBankPlan::predicted`), its landing launch
/// (`routed_land`), after its park, and where the layer has a CPU site
/// its request (`routed_site_request`), before its publication.
fn route(
    step: &mut Step<'_>,
    load: &ModelLoadPlan,
    scope: WeightScope,
    resident: Kernel,
    streamed: Kernel,
    work: Work,
) -> Result<(), AssessmentError> {
    let Some((bank, index)) = streamed::served(load, scope) else {
        return step.call(resident, work);
    };
    if bank.sets_step(index) {
        step.call(Kernel::of::<kernels::routed_step::Entry>(), Work::default())?;
    }
    let layers = bank.layers();
    let hinted = bank
        .predicted(index)
        .flat_map(|target| {
            let layer = &layers[target];
            layer.hinted.iter().filter_map(move |hint| hint.element.map(|element| (layer.scope, hint.kind, element)))
        })
        .map(|(scope, kind, element)| planned(load, scope, kind, element))
        .collect::<Result<Vec<_>, _>>()?;
    step.call_with(streamed, &[("AHEAD", bank.ahead(index))], join([work, projection(&hinted)?])?)?;
    step.park();
    // The landing launch converts the bank's records, from their stored
    // element into their resident one: its operands are the records', not a
    // weight the step reads.
    let record = bank.binding(index).map_err(demand_error)?.record;
    let recipe = operand_recipe(record.iter().flat_map(|(_, tensor)| [tensor.source, tensor.resident]));
    step.call_as(Kernel::of::<kernels::routed_land::Entry>(), recipe, &[], &[], Work::default())?;
    if bank.sited(index) {
        step.call(Kernel::of::<kernels::routed_site_request::Entry>(), Work::default())?;
        step.park();
    }
    Ok(())
}

/// The combine of the routed layer of `scope` doing `work`: of a layer
/// with a CPU site, its streamed form's projections (`project`) and their
/// sum (`sited`) after its park, which waits for the CPU's answer; of any
/// other, `resident`. Each reads its choices by the layer's sources.
fn combine(
    step: &mut Step<'_>,
    load: &ModelLoadPlan,
    scope: WeightScope,
    [resident, project, sited]: [Kernel; 3],
    work: Work,
) -> Result<(), AssessmentError> {
    let sources = sources(load, scope);
    match streamed::served(load, scope).filter(|(bank, index)| bank.sited(*index)) {
        None => step.call_with(resident, &sources, work),
        Some(_) => {
            step.call_with(project, &sources, work)?;
            step.park();
            step.call(sited, Work::default())
        }
    }
}

/// `operators::routed::graph` at decode rows in `scope`: the shared
/// expert's expansion (`dense_expand`, or `dense_up` up-only) and down
/// projection onto the sum's root (`dense_output`), selection
/// (`routed_select`), the latent projections (`project_rows`,
/// `dense_output`), and the selected experts' expansion (`routed_gate_up`,
/// or `routed_up` up-only) and down projection (`routed_down`), a streamed
/// layer's selection and down in its streamed form ([`route`],
/// [`combine`]). Its input is the caller's.
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
    route(
        step,
        load,
        scope,
        Kernel::of::<kernels::routed_select::Entry>(),
        Kernel::of::<kernels::routed_select_streamed::Entry>(),
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
    let sources = sources(load, scope);
    step.call_with(
        expansion,
        &sources,
        Work {
            experts: Some(chosen(&tensors)?),
            ..Work::default()
        },
    )?;
    combine(
        step,
        load,
        scope,
        [
            Kernel::of::<kernels::routed_down::Entry>(),
            Kernel::of::<kernels::routed_down_project::Entry>(),
            Kernel::of::<kernels::routed_down_sited::Entry>(),
        ],
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
    let indexer = LayerRef::IndexKeys {
        block: layer,
        part: 0,
    };
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

/// `left · right`, its overflow named by `what`.
fn product(left: u64, right: u64, what: &str) -> Result<u64, AssessmentError> {
    left.checked_mul(right)
        .ok_or_else(|| demand_error(format!("{what} overflow")))
}

/// The bytes of one row of `layer`'s component in the Block domain holding
/// it (a pooled source's rows, or an indexer's keys over them), and the
/// domain's rate: the tokens one of its rows covers.
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

/// The pooled rows part `part` of target block `layer`'s index-list source
/// reads: its own (`LayerRef::Pooled`), or those of the layer a Shared
/// domain names as their source (a Reindex or reused source's).
fn picked_rows(
    domains: &[HistoryDomainLayout],
    layer: u32,
    part: u32,
) -> Result<LayerRef, AssessmentError> {
    let ours = LayerRef::Pooled { block: layer, part };
    domains
        .iter()
        .find_map(|domain| match domain {
            HistoryDomainLayout::Shared { source, layers } => {
                layers.contains(&ours).then_some(*source)
            }
            HistoryDomainLayout::Block { components, .. } => components
                .iter()
                .any(|component| component.layer == ours)
                .then_some(ours),
            HistoryDomainLayout::Token { .. } | HistoryDomainLayout::Window { .. } => None,
        })
        .ok_or_else(|| demand_error(format!("attention layer {layer} has no pooled rows")))
}

/// The calls of target block `layer`'s index-list source `picked` before
/// attention reads it (`operators::attention::pooled::picked`), and what
/// that read adds to attention's: each row's `picks` rows of the source's
/// plane, each scored and accumulated as a history token is
/// (`flops_per_pick`). A reused source launches nothing. An own source makes
/// its rows (`own_rows`): the `u | g` projection, `u`'s widening where the
/// pool is ungated, then `pool_rows`, which reads its incomplete group's
/// pending rows and writes the plane row it completes, or at rate 1
/// `attention_append_dense`, which writes each token's row; and its
/// indexer writes their keys (`selection::indexer`: the latent's key
/// projection and `indexer_keys`). An own or Reindex indexer projects its
/// query and head weights, scores each unit of its list (`indexer_score`,
/// reading its key and writing its F32 score, `2·H·W` FLOPs) and picks
/// (`rows_topk`, reading the scores); a producing indexer then scores the
/// pool's blocks (`candidate_blocks`), picks them (`rows_topk`) and lists
/// their rows (`candidate_rows`, over a row's few blocks). A list is every
/// complete unit, or a restricted indexer's pool of at most its units. The
/// fused projections' input norms are priced as attention's are: by their
/// consumer, not read as weights. The picks' products are in the
/// attention's `activation` element, as its history's.
fn picked<'a>(
    step: &mut Step<'_>,
    domains: &[HistoryDomainLayout],
    layer: u32,
    picked: PickedBinding,
    (picks, flops_per_pick): (u64, u64),
    activation: Element,
    weight: impl Fn(WeightKind, Element) -> Result<&'a WeightPlan, AssessmentError>,
) -> Result<Work, AssessmentError> {
    let project = Kernel::of::<kernels::attention_project::Entry>();
    let plane_rows = picked_rows(domains, layer, picked.part())?;
    let (plane_row, _) = block_component(domains, plane_rows)?;
    let indexer = match picked {
        PickedBinding::Reused => None,
        PickedBinding::Own(pooled) => {
            let mut projected = vec![weight(WeightKind::PooledProjection, pooled.projection)?];
            if let Some(gate) = pooled.gate {
                projected.push(weight(WeightKind::PooledGate, gate)?);
            }
            step.call(project, projection(&projected)?)?;
            if pooled.gate.is_none() {
                step.call(Kernel::of::<kernels::widen_rows::Entry>(), Work::default())?;
            }
            match Pooling::of(pooled.rate) {
                Pooling::Grouped => {
                    let pending = domains
                        .iter()
                        .filter(|domain| matches!(domain, HistoryDomainLayout::Window { .. }))
                        .find_map(|domain| component_row_bytes(domain, plane_rows).transpose())
                        .transpose()?
                        .ok_or_else(|| {
                            demand_error(format!("pooled layer {layer} has no pending group"))
                        })?;
                    let local = product(pending, pooled.rate, "pooled row bytes")?
                        .checked_add(plane_row)
                        .ok_or_else(|| demand_error("pooled row bytes overflow"))?;
                    step.call(Kernel::of::<kernels::pool_rows::Entry>(), rows(local))?;
                }
                Pooling::Identity => step.call(
                    Kernel::of::<kernels::attention_append_dense::Entry>(),
                    rows(plane_row),
                )?,
            }
            let keys = LayerRef::IndexKeys {
                block: layer,
                part: pooled.part,
            };
            let (key_row, _) = block_component(domains, keys)?;
            step.call(
                project,
                projection(&[weight(WeightKind::IndexerKey, pooled.key)?])?,
            )?;
            step.call(Kernel::of::<kernels::indexer_keys::Entry>(), rows(key_row))?;
            Some((pooled.indexer, keys))
        }
        PickedBinding::Reindex {
            source,
            part,
            indexer,
        } => Some((indexer, LayerRef::IndexKeys { block: source, part })),
    };
    if let Some((indexer, keys)) = indexer {
        let (key_row, rate) = block_component(domains, keys)?;
        step.call(
            project,
            projection(&[weight(WeightKind::IndexerQuery, indexer.query)?])?,
        )?;
        if let Some(head_weights) = indexer.head_weights {
            step.call(
                project,
                projection(&[weight(WeightKind::IndexerHeadWeights, head_weights)?])?,
            )?;
        }
        let window = match indexer.candidates {
            CandidatesBinding::All | CandidatesBinding::Produce(_) => None,
            CandidatesBinding::Restrict(pool) => {
                Some(product(pool.units(), rate, "candidate pool tokens")?)
            }
        };
        // Traffic and arithmetic over each `tokens` tokens of context.
        let read = |window, bytes: u64, flops: u64, tokens: u64| Work {
            history: Some(HistoryRead {
                layout: None,
                bytes_per_token: bytes.div_ceil(tokens),
                flops_per_token: flops.div_ceil(tokens),
                window,
            }),
            ..Work::default()
        };
        let score = product(2 * indexer.heads, indexer.width, "indexer score FLOPs")?;
        step.call(
            Kernel::of::<kernels::indexer_score::Entry>(),
            read(window, key_row + 4, score, rate),
        )?;
        let topk = Kernel::of::<kernels::rows_topk::Entry>();
        step.call(topk, read(window, 4, 0, rate))?;
        if let CandidatesBinding::Produce(pool) = indexer.candidates {
            // A block's scores, then its score and its span (two i32).
            let block = product(pool.block_units, rate, "candidate block tokens")?;
            let block_units = product(4, pool.block_units, "candidate block bytes")?;
            step.call(
                Kernel::of::<kernels::candidate_blocks::Entry>(),
                read(None, block_units + 4 + 8, 0, block),
            )?;
            step.call(topk, read(None, 4 + 8, 0, block))?;
            step.call(Kernel::of::<kernels::candidate_rows::Entry>(), Work::default())?;
        }
    }
    Ok(Work {
        row_bytes: product(picks, plane_row, "picked row bytes")?,
        row_flops: vec![Flops {
            format: activation,
            flops: product(picks, flops_per_pick, "picked row FLOPs")?,
        }],
        ..Work::default()
    })
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
            words: match load.expert_bank() {
                Some(_) => STREAMED,
                None => &[],
            },
            resolved: HashMap::new(),
            calls: Vec::new(),
            boundaries: 0,
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
            let streams = block.block_streams();
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
                    let query = if shape.form.gate == crate::planning::GateForm::Interleaved {
                        WeightKind::QueryGate
                    } else {
                        WeightKind::Query
                    };
                    // The fused projection's query segment: the query's, or
                    // a low-rank query's `down` (`query_segment_kind`).
                    let segment = match binding.low_rank_query {
                        None => query,
                        Some(_) => WeightKind::QueryDown,
                    };
                    let projections = [
                        (true, segment, binding.query_segment()),
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
                    // A low-rank query's `qr` rows, widened to F32
                    // (`widen_rows`), then projected to the query heads
                    // through its up weight (`binding.query`) under its norm.
                    if binding.low_rank_query.is_some() {
                        step.call(Kernel::of::<kernels::widen_rows::Entry>(), Work::default())?;
                        step.call(project, projection(&[weight(query, binding.query)?])?)?;
                    }
                    // The entry attending the layer's history, and the
                    // layout it reads: the layer's own (`history_codec`; the
                    // model's own store is dense whatever the policy).
                    let (attend, history_layout) = match binding.history.encoding() {
                        // The entry the graph attends decode rows with
                        // (`graph::dense_decodes`): the prefill entry where
                        // the decode entry holds no form of the shape.
                        HistoryEncoding::Dense => (
                            if crate::operators::attention::graph::dense_decodes(backend, binding)
                                .map_err(|error| demand_error(format!("{error:?}")))?
                            {
                                Kernel::of::<kernels::attention_decode::Entry>()
                            } else {
                                Kernel::of::<kernels::attention_prefill::Entry>()
                            },
                            KvCodec::Dense,
                        ),
                        HistoryEncoding::Affine => (
                            Kernel::of::<kernels::attention_decode_k8v4::Entry>(),
                            KvCodec::AffineK8V4,
                        ),
                        HistoryEncoding::Unsupported => {
                            return Err(demand_error(
                                "rotated K4/V4 history has no native attention entry",
                            ));
                        }
                    };
                    // The load prepares the attention entries per layer
                    // form (a compressed layer's picks launch where a
                    // window layer has none): the call is its layer's.
                    let statics =
                        crate::operators::attention::graph::form_statics(&shape, binding.history)
                            .map_err(demand_error)?;
                    let (row_bytes, window) =
                        history_read(&layout.target_history, LayerRef::Target(layer))?;
                    let widths = shape.widths;
                    let flops_per_token = 2 * widths.kv_heads * widths.group * (2 * widths.width);
                    let read = |window| HistoryRead {
                        layout: Some(history_layout),
                        bytes_per_token: row_bytes,
                        flops_per_token,
                        window,
                    };
                    // An index-list source's calls, and the picks attention
                    // reads of its plane beside its history.
                    let picks = match binding.picked {
                        None => Work::default(),
                        Some(source) => picked(
                            &mut step,
                            &layout.target_history,
                            layer,
                            source,
                            (shape.widths.picks, flops_per_token),
                            binding.activation,
                            weight,
                        )?,
                    };
                    match binding.selection {
                        None => step.call_specialized(
                            attend,
                            &statics,
                            &[],
                            Work {
                                history: Some(read(window)),
                                ..picks
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
                            step.call_specialized(
                                attend,
                                &statics,
                                &[],
                                Work {
                                    history: Some(read(Some(
                                        window.map_or(selected, |window| window.min(selected)),
                                    ))),
                                    ..picks
                                },
                            )?;
                        }
                    }
                    // A grouped output's first product, each group's run of
                    // the attended row through its block of `a`
                    // (`project_rows`), which the output projection reads.
                    if let Some(grouped) = binding.grouped {
                        step.call(
                            Kernel::of::<kernels::project_rows::Entry>(),
                            projection(&[weight(
                                WeightKind::AttentionOutputGrouped,
                                grouped.element,
                            )?])?,
                        )?;
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
                // expert's down projections (`routed_output`). A streamed
                // layer routes and combines in its streamed form ([`route`],
                // [`combine`]).
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
                    let sources = sources(load, scope);
                    match DecodeForm::of(backend).map_err(demand_error)? {
                        DecodeForm::Expand => {
                            streamed::expand_form(streamed::streams(load, scope)).map_err(demand_error)?;
                            step.call(Kernel::of::<kernels::routed_route::Entry>(), router)?;
                            step.call(
                                Kernel::of::<kernels::routed_expand::Entry>(),
                                join([choices, shared])?,
                            )?;
                        }
                        DecodeForm::SharedRoute => {
                            route(
                                &mut step,
                                load,
                                scope,
                                Kernel::of::<kernels::routed_route_shared::Entry>(),
                                Kernel::of::<kernels::routed_route_streamed::Entry>(),
                                join([router, shared])?,
                            )?;
                            step.call_with(
                                Kernel::of::<kernels::routed_gate_up::Entry>(),
                                &sources,
                                choices,
                            )?;
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
                    combine(
                        &mut step,
                        load,
                        scope,
                        [
                            Kernel::of::<kernels::routed_output::Entry>(),
                            Kernel::of::<kernels::routed_output_project::Entry>(),
                            Kernel::of::<kernels::routed_output_sited::Entry>(),
                        ],
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
                    // Over Sinkhorn streams the routed branch joins the
                    // streams `[S·H]` through `stream_join`, from its
                    // branch row `[H]`.
                    if let Some(BlockStreams::Sinkhorn(SinkhornBinding { hidden, streams, .. })) = streams {
                        step.call(
                            Kernel::of::<kernels::stream_join::Entry>(),
                            rows(f32_bytes(2 * streams * hidden + hidden)?),
                        )?;
                    }
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
                recurrent_component += binding.bank_components();
            }
        }

        // The last block's graph ends with the decoder's exit: a low-rank
        // exit site, which injects into nothing, or the collapse of Sinkhorn
        // streams `[S·H]` into one row `[H]` under the last site's pre.
        match target.blocks().last().and_then(|block| block.block_streams()) {
            Some(BlockStreams::LowRank(streams)) => {
                if let Some(exit) = streams.exit {
                    mix_site(&mut step, load, WeightScope::Target, exit, None)?;
                }
            }
            Some(BlockStreams::Sinkhorn(SinkhornBinding { hidden, streams, .. })) => step.call(
                Kernel::of::<kernels::stream_collapse::Entry>(),
                rows(f32_bytes(streams * hidden + hidden)?),
            )?,
            None => {}
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
        Ok(Self { calls: step.calls, boundaries: step.boundaries })
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
        let definition = crate::planning::fixtures::fixture_definition();
        let manifest = crate::planning::fixtures::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows8,
            crate::PlannedMethod::Plain,
            crate::planning::fixtures::LIMITS,
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
        let definition = crate::planning::fixtures::hyper_connection_fixture_definition();
        let manifest = crate::planning::fixtures::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows8,
            crate::PlannedMethod::Plain,
            crate::planning::fixtures::LIMITS,
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

    /// Over Sinkhorn streams each sublayer reads through its site
    /// (`stream_mix_sinkhorn`: its F32 function, base and scale, the
    /// streams and pre read and the input, pre and join values written, its
    /// projection's FLOPs and its 20 normalizations' of the comb), the
    /// general routed branch joins through `stream_join`, and the exit
    /// collapses the streams (`stream_collapse`). The low-rank decoder's
    /// demand is unchanged (`a_streams_decode_step_streams_its_mix_sites_and_no_norms`).
    #[test]
    fn a_sinkhorn_decode_step_bills_its_sites_joins_and_collapse() {
        let definition = crate::planning::fixtures::sinkhorn_fixture_definition(2);
        let manifest = crate::planning::fixtures::fixture_manifest(&definition);
        let selection = crate::ComponentSelection { head: false, vision: false };
        let limits = crate::planning::fixtures::LIMITS;
        let load =
            ModelLoadPlan::derive(&manifest, &definition, selection, Layout::Rows8, crate::PlannedMethod::Plain, limits)
                .unwrap();
        let demand = demand(&definition, &load, KvCodec::Dense).unwrap();
        let block = [
            "stream_mix_sinkhorn",
            "attention_project",
            "attention_decode",
            "attention_output",
            "stream_mix_sinkhorn",
            "routed_select",
            "routed_gate_up",
            "routed_down",
            "stream_join",
        ];
        let expected = ["embedding_rows", "repeat_rows"]
            .into_iter()
            .chain(block)
            .chain(block)
            .chain(["stream_collapse", "readout_features_rows", "readout_head_rows", "sample_rows"])
            .collect::<Vec<_>>();
        let names = demand.calls.iter().map(|call| call.launches.kernel().name()).collect::<Vec<_>>();
        assert_eq!(names, expected);
        let (s, h, iterations) = (4, 128, 20);
        let call = |name| demand.calls.iter().find(|call| call.launches.kernel().name() == name).unwrap();
        let site = &call("stream_mix_sinkhorn").work;
        let mixer = WeightScope::TargetSublayer(SublayerIndex { block: 0, sublayer: 0 });
        let bytes = |kind| {
            load.target().iter().find(|weight| weight.role.scope == mixer && weight.role.kind == kind).unwrap().resident_bytes
        };
        let site_weights = [WeightKind::HyperConnectionFunction, WeightKind::HyperConnectionBase, WeightKind::HyperConnectionScale]
            .map(bytes)
            .iter()
            .sum::<u64>();
        assert_eq!(site.weights, vec![Bytes { format: Element::f32(), bytes: site_weights }]);
        assert_eq!(site.row_bytes, 4 * (s * h + h + 2 * s + s * s));
        let projection = 2 * (2 + s) * s * s * h;
        let normalizations = 2 * iterations - 1;
        assert_eq!(site.row_flops, vec![Flops { format: Element::f32(), flops: projection + normalizations * 2 * s * s }]);
        assert_eq!(call("stream_join").work.row_bytes, 4 * (2 * s * h + h));
        assert_eq!(call("stream_collapse").work.row_bytes, 4 * (s * h + h));
    }

    /// An n-gram input charges its row rule over its token window, its
    /// heads' stored rows, its key and value projections, its gate's norms,
    /// and its row op's norm, taps and window; the recurrent mixers after it
    /// read their own bank components.
    #[test]
    fn an_ngram_input_streams_its_rows_projections_and_windows() {
        let load_of = |definition: &ModelDefinition| {
            let manifest = crate::planning::fixtures::stored_table_manifest(definition);
            ModelLoadPlan::derive(
                &manifest,
                definition,
                crate::ComponentSelection {
                    head: false,
                    vision: false,
                },
                Layout::Rows8,
                crate::PlannedMethod::Plain,
                crate::planning::fixtures::LIMITS,
            )
            .unwrap()
        };
        let definition = crate::planning::fixtures::qwen4exp_fixture_definition();
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
        let ngram = heads * table.resident_bytes / table.table_slots()
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
        // `ngram_inject`'s row reads its streams, its 2S scales and its
        // value row, and writes its streams, all F32.
        let magnitude_family_contracts::ResidualForm::HyperConnections(residual) =
            &definition.decoder.residual
        else {
            panic!("the qwen4exp fixture keeps hyper-connection streams");
        };
        let (s, h) = (residual.streams, definition.decoder.hidden);
        let inject_rows = 4 * (2 * s * h + 2 * s + h);
        assert_eq!(streamed(&with), streamed(&without) + ngram + inject_rows);
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
            Err(AssessmentError::Catalog(crate::CatalogError {
                failure: crate::CatalogFailure::Preparation { entry: "attention_decode", .. },
                ..
            }))
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
            HeadNorm, HeadWeights, HistoryReads, Indexer, IndexerInput, IndexerKeys,
            IndexerQuery, Operator, Pool, PooledRows, RmsNorm, RowRounding, Selection,
            WeightDescriptor,
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
        attention.history[0].reads = HistoryReads::Selected(Selection::Own(Box::new(Indexer {
            heads: 2,
            width: 64,
            query: IndexerQuery {
                input: IndexerInput::Hidden,
                projection: WeightDescriptor::stored("indexer_q", vec![128, hidden]),
                norm: HeadNorm::Rms(norm("indexer_q_norm")),
                rounding: RowRounding::None,
            },
            keys: IndexerKeys::Pooled(PooledRows {
                rate: 4,
                projection: WeightDescriptor::stored("indexer_k", vec![64, hidden]),
                pool: Pool::Mean,
                norm: norm("indexer_k_norm"),
                rounding: RowRounding::None,
            }),
            head_weights: HeadWeights::Uniform,
            scale: 0.125,
            top: 8,
            candidates: magnitude_family_contracts::Candidates::All,
        })));
        let manifest = crate::planning::fixtures::fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            Layout::Rows8,
            crate::PlannedMethod::Plain,
            crate::planning::fixtures::LIMITS,
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
        assert!(experts(&[&tensor], 8, 9).is_err());
        // A banked stack (its group's 20 records, charged its 3 slots)
        // reads one record a slab, as resident.
        let mut banked = tensor.clone();
        banked.shape[0] = 20;
        banked.resident_bytes = 3 * 32 * 64 * 2;
        assert_eq!(experts(&[&banked], 8, 2).unwrap(), experts(&[&tensor], 8, 2).unwrap());
    }
}
