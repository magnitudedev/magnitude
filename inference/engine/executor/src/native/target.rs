use crate::operators::streams::{BlockStreams, MixDownBinding, MixUpBinding, SinkhornBinding, Sites};
use crate::planning::RecordTensors;
use crate::{
    AttentionBinding, DenseBinding, EmbeddingBinding, FeaturesBinding, GeneralRoutedBinding,
    ReadoutBinding, RecurrentBinding, RoutedBinding, ShortConvBinding, StateSpaceBinding,
};
use magnitude_kernels::{
    attention_append_dense, attention_append_k8v4, attention_decode, attention_decode_k8v4,
    attention_output, attention_prefill, attention_prefill_k8v4, attention_project, dense_expand,
    dense_output, embedding_rows, gated_delta_chunk, gated_delta_project,
    gated_delta_project_convolved, gated_delta_step, gated_delta_step_convolved, indexer_select,
    post_norm_residual, project_rows, readout_exact_rows, readout_features_rows, readout_head_rows,
    readout_planes_rows, readout_refine_rows, readout_selected_rows, readout_top_rows,
    routed_combine, routed_expand, routed_experts, routed_group, routed_output, routed_route,
    routed_route_shared,
};
use magnitude_kernels::{
    dense_up, feature_rows, routed_down, routed_experts_up, routed_gate_up, routed_scatter,
    routed_select, routed_up, state_space_chunk, state_space_gate, state_space_step, tap_rows,
    widen_rows,
};
use magnitude_kernels::{short_conv_project, short_conv_rows};
use magnitude_kernels::{
    candidate_blocks, candidate_rows, indexer_keys, indexer_score, pool_rows, rows_topk,
};
use seismic::NativeKernel;
use std::collections::HashMap;

/// Immutable decoder-stage specializations keyed by semantic bindings.
#[derive(Debug, Default)]
pub struct TargetKernels {
    pub(super) residual_features: Option<NativeKernel<feature_rows::Entry>>,
    pub(super) embedding: HashMap<EmbeddingBinding, NativeKernel<embedding_rows::Entry>>,
    pub(super) attention: HashMap<AttentionBinding, AttentionKernels>,
    pub(super) recurrent: HashMap<RecurrentBinding, RecurrentKernels>,
    pub(super) state_space: HashMap<StateSpaceBinding, StateSpaceKernels>,
    pub(super) short_conv: HashMap<ShortConvBinding, ShortConvKernels>,
    pub(super) general_routed: HashMap<GeneralRoutedBinding, GeneralRoutedKernels>,
    pub(super) dense: HashMap<DenseBinding, DenseKernels>,
    pub(super) routed: HashMap<RoutedBinding, RoutedKernels>,
    pub(super) readout: HashMap<ReadoutBinding, ReadoutKernels>,
    pub(super) features: HashMap<FeaturesBinding, NativeKernel<readout_features_rows::Entry>>,
    pub(super) taps: Option<TapKernels>,
    pub(super) per_layer_entry: HashMap<crate::PerLayerEntryBinding, PerLayerEntryKernels>,
    pub(super) per_layer: HashMap<crate::PerLayerBinding, PerLayerKernels>,
    pub(super) parallel: HashMap<crate::ParallelBinding, ParallelKernels>,
    pub(super) streams: StreamsEntryKernels,
    pub(super) ngram: HashMap<crate::NGramBinding, NGramKernels>,
    /// The bank entries of a routed layer whose experts are streamed, by
    /// its feed-forward's binding (either routed form) and its bank's.
    pub(super) banks: HashMap<(crate::planning::BankedFeedForward, crate::BankBinding), ExpertBankKernels>,
}

/// A streamed layer's decode routes, one by `AHEAD`: without hints (the
/// last streamed layer), then at each count of leads up to `HINT_LEADS`
/// (`ExpertBankPlan::ahead`: the layer before the last has one).
pub(crate) const LEADS: usize = crate::planning::HINT_LEADS as usize + 1;

/// A streamed layer's decode route and resolve by its routed form, each by
/// `AHEAD` ([`LEADS`]): a lead's kernel where a layer of the binding
/// launches it, none where none does (a binding without the last streamed
/// layer has no `AHEAD` 0 launch, one of only that layer no other).
#[derive(Clone, Debug)]
pub enum BankRouteKernels {
    /// The fused form's routing and shared product (`routed_route_shared`'s).
    Fused([Option<NativeKernel<magnitude_kernels::routed_route_streamed::Entry>>; LEADS]),
    /// The general form's selection (`routed_select`'s).
    General([Option<NativeKernel<magnitude_kernels::routed_select_streamed::Entry>>; LEADS]),
}

/// The routed form whose decode route a bank's entries take.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum BankRouteForm {
    Fused,
    General,
}

/// A streamed routed layer's bank entries (`operators::routed::streamed`):
/// decode rows route, resolve and assign, then wait for the experts the
/// host lands; larger rows set the step,
/// request, and wait and convert a round at a time. `convert` and `seed`
/// are per routed tensor of the record.
#[derive(Clone, Debug)]
pub struct ExpertBankKernels {
    pub route: BankRouteKernels,
    pub tier_wait: NativeKernel<magnitude_kernels::routed_tier_wait::Entry>,
    pub step: NativeKernel<magnitude_kernels::routed_step::Entry>,
    pub scratch_request: NativeKernel<magnitude_kernels::routed_scratch_request::Entry>,
    pub scratch_wait: NativeKernel<magnitude_kernels::routed_scratch_wait::Entry>,
    pub scratch_grids: NativeKernel<magnitude_kernels::routed_scratch_grids::Entry>,
    /// A round's convert list staged into device memory, which its converts
    /// read.
    pub list_stage: NativeKernel<magnitude_kernels::routed_list_stage::Entry>,
    /// A merge set's pass's map and step resolved into device memory,
    /// which its expert entries read.
    pub pass_map: NativeKernel<magnitude_kernels::routed_pass_map::Entry>,
    pub convert: RecordTensors<NativeKernel<magnitude_kernels::routed_convert::Entry>>,
    /// The device's placements into the bank's records, whoever places
    /// (`BankBinding::placer`): a prompt group's seeds, copied from its
    /// scratch slots.
    pub placement: DevicePlacement,
    /// A shrink's moves of the bank's kept records, which the device runs in
    /// queue order ahead of every later step, whoever places.
    pub moves: BankMoves,
    /// Its CPU site, where the bank has a host tier the CPU reads its
    /// misses from, the backend serves the site's entries, and, once
    /// qualified, the precision system admits its arithmetic
    /// ([`BankSite::admitted`]).
    pub site: Option<BankSite>,
}

/// A streamed layer's CPU site entries (streaming_words.seismic's request
/// and answer): its request (`routed_site_request`), after the route or its
/// tier wait and before gate/up, and its combine in its routed form (the
/// GPU's projections, then the sum waiting for the CPU's answer); and the
/// form the site's pool computes in.
#[derive(Clone, Debug)]
pub struct BankSite {
    pub request: NativeKernel<magnitude_kernels::routed_site_request::Entry>,
    pub combine: BankCombine,
    pub form: SiteForm,
    /// What the precision system admitted of the site, as the measurement
    /// job stored it; none until it has (`SiteRequest`), when the site is
    /// held but serves nothing.
    pub admitted: Option<SiteAdmission>,
}

/// A CPU site's stored admission (`SiteRequest::verdict`): the precision
/// system admitted the site's arithmetic against the reference, f32 math
/// over the stored weights.
#[derive(Clone, Copy, Debug)]
pub struct SiteAdmission;

impl BankSite {
    /// Whether a streamed layer of feed-forward `slot` whose bank is `bank`
    /// has a CPU site on `backend`: the bank has a host tier (across a link;
    /// the route assigns its misses between the CPU and the link each step)
    /// and the backend serves the site's request and the combine of the
    /// layer's form. The one rule preparation and the graph's checked
    /// entries share.
    pub(crate) fn served(
        slot: crate::planning::BankedFeedForward,
        bank: crate::BankBinding,
        backend: seismic::BackendName,
    ) -> Result<bool, String> {
        use crate::planning::BankedFeedForward;
        use magnitude_kernels::{
            routed_down_project, routed_down_sited, routed_output_project, routed_output_sited, routed_site_request,
        };
        fn declares<E: seismic::Entry>(backend: seismic::BackendName) -> Result<bool, String> {
            seismic::generated::native_implementation_for_backend::<E>(backend)
                .map(|native| native.is_some())
                .map_err(|error| error.to_string())
        }
        let combine = match slot {
            BankedFeedForward::Routed(_) => {
                declares::<routed_output_project::Entry>(backend)? && declares::<routed_output_sited::Entry>(backend)?
            }
            BankedFeedForward::GeneralRouted(_) | BankedFeedForward::Parallel(_) => {
                declares::<routed_down_project::Entry>(backend)? && declares::<routed_down_sited::Entry>(backend)?
            }
        };
        // The CPU computes a miss from its host tier record; unified memory
        // has no tier and no CPU site.
        Ok(bank.tier > 0 && combine && declares::<routed_site_request::Entry>(backend)?)
    }
}

/// A streamed layer's combine with the CPU's answer, by its routed form:
/// the GPU's projections (`project`, which run while the CPU computes its
/// misses), then their sum waiting for the CPU's answer (`sited`): the fused form's output (`routed_output_project`,
/// `routed_output_sited`) or the general form's down (`routed_down_project`,
/// `routed_down_sited`).
#[derive(Clone, Debug)]
pub enum BankCombine {
    Fused {
        project: NativeKernel<magnitude_kernels::routed_output_project::Entry>,
        sited: NativeKernel<magnitude_kernels::routed_output_sited::Entry>,
    },
    General {
        project: NativeKernel<magnitude_kernels::routed_down_project::Entry>,
        sited: NativeKernel<magnitude_kernels::routed_down_sited::Entry>,
    },
}

/// What the CPU site's pool computes a layer's choices in: the element its
/// gate/up products are published in (`A`), its expansion's function (the
/// decode entries' `activation`) and clamp (their LB).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SiteForm {
    pub activation: seismic::Element,
    pub function: i32,
    pub limit: Option<magnitude_family_contracts::GluLimit>,
}

impl SiteForm {
    /// The form a layer of feed-forward `slot` computes in.
    pub(crate) fn of(slot: crate::planning::BankedFeedForward) -> Self {
        use crate::planning::BankedFeedForward;
        match slot {
            BankedFeedForward::Routed(binding) => Self {
                activation: binding.activation,
                function: crate::operators::routed::FUSED_ACTIVATION,
                // The fused form's experts are unclamped (`is_fused`).
                limit: None,
            },
            BankedFeedForward::GeneralRouted(binding)
            | BankedFeedForward::Parallel(crate::ParallelBinding { routed: binding, .. }) => Self {
                activation: binding.activation,
                function: binding.shape.experts_expansion.activation,
                limit: binding.shape.experts_expansion.limit,
            },
        }
    }
}

/// The device's copies into a bank's records: a prompt group's seeding
/// (each routed tensor's copy from the scratch tier into the bank's records
/// on the seed list's grid, then the commit of each seeded record, which
/// the host then maps).
#[derive(Clone, Debug)]
pub struct DevicePlacement {
    pub seed: RecordTensors<NativeKernel<magnitude_kernels::routed_copy::Entry>>,
    pub seed_commit: NativeKernel<magnitude_kernels::routed_seed_commit::Entry>,
    /// The launch grid of the seed list's entries, which the seed copy
    /// launches on.
    pub list_grid: NativeKernel<magnitude_kernels::routed_list_grid::Entry>,
}

/// A shrink's record moves (`routed_move`).
#[derive(Clone, Debug)]
pub struct BankMoves {
    /// Each routed tensor's records.
    pub records: RecordTensors<NativeKernel<magnitude_kernels::routed_move::Entry>>,
}

impl ExpertBankKernels {
    /// Whether `backend` declares every entry a bank of `form` prepares
    /// (`Preparation::bank`, one field of this struct each).
    pub(crate) fn declared(
        backend: seismic::BackendName,
        form: BankRouteForm,
    ) -> Result<bool, seismic::CheckedBundleError> {
        use magnitude_kernels::{
            routed_convert, routed_copy, routed_list_grid, routed_list_stage, routed_move, routed_pass_map, routed_route_streamed,
            routed_scratch_grids, routed_scratch_request, routed_scratch_wait, routed_seed_commit,
            routed_select_streamed, routed_step, routed_tier_wait,
        };
        type Declares = fn(seismic::BackendName) -> Result<bool, seismic::CheckedBundleError>;
        fn declares<E: seismic::Entry>(backend: seismic::BackendName) -> Result<bool, seismic::CheckedBundleError> {
            seismic::generated::native_implementation_for_backend::<E>(backend).map(|native| native.is_some())
        }
        let route: Declares = match form {
            BankRouteForm::Fused => declares::<routed_route_streamed::Entry>,
            BankRouteForm::General => declares::<routed_select_streamed::Entry>,
        };
        let bank: [Declares; 13] = [
            route,
            declares::<routed_tier_wait::Entry>,
            declares::<routed_step::Entry>,
            declares::<routed_scratch_request::Entry>,
            declares::<routed_scratch_wait::Entry>,
            declares::<routed_scratch_grids::Entry>,
            declares::<routed_convert::Entry>,
            declares::<routed_copy::Entry>,
            declares::<routed_seed_commit::Entry>,
            declares::<routed_move::Entry>,
            declares::<routed_list_grid::Entry>,
            declares::<routed_list_stage::Entry>,
            declares::<routed_pass_map::Entry>,
        ];
        for declares in bank.iter() {
            if !declares(backend)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// An n-gram input sublayer (`operators::ngram`): the table rows of each
/// position (`rows`), read from the streamed table and decoded to
/// activations (`stored`), their key and value projections into F32, the
/// gate and the row op.
#[derive(Clone, Debug)]
pub struct NGramKernels {
    pub rows: NativeKernel<magnitude_kernels::ngram_rows::Entry>,
    pub stored: NativeKernel<magnitude_kernels::stored_rows::Entry>,
    pub key: NativeKernel<project_rows::Entry>,
    pub value: NativeKernel<project_rows::Entry>,
    pub gate: NativeKernel<magnitude_kernels::ngram_gate::Entry>,
    pub output: NGramOutputKernel,
}

/// An n-gram input's row op: the gated value added into the streams, or
/// added with its convolution over the window.
#[derive(Clone, Debug)]
pub enum NGramOutputKernel {
    Add(NativeKernel<magnitude_kernels::ngram_add::Entry>),
    Inject(NativeKernel<magnitude_kernels::ngram_inject::Entry>),
}

/// A mix site's entries (`operators::streams`): its first stage (`down`,
/// with inject values at a sublayer's site, without at the exit) and its
/// second (`up`).
#[derive(Clone, Debug)]
pub struct MixSiteKernels {
    pub down: NativeKernel<magnitude_kernels::stream_mix_down::Entry>,
    pub up: NativeKernel<magnitude_kernels::stream_mix_up::Entry>,
}

/// The entries of Sinkhorn streams (`operators::streams`): every site's
/// (`stream_mix_sinkhorn`), the exit's collapse, and the join of a
/// sublayer whose output entry joins nothing itself.
#[derive(Clone, Debug)]
pub struct SinkhornKernels {
    pub site: NativeKernel<magnitude_kernels::stream_mix_sinkhorn::Entry>,
    pub collapse: NativeKernel<magnitude_kernels::stream_collapse::Entry>,
    pub join: NativeKernel<magnitude_kernels::stream_join::Entry>,
}

/// A block's mix sites' entries, by its streams' mixing: each low-rank
/// site's, at that site's own binding, or the Sinkhorn entries.
#[derive(Clone, Debug)]
pub enum MixingKernels {
    LowRank(Sites<MixSiteKernels>),
    Sinkhorn(SinkhornKernels),
}

/// A block's hyper-connection streams (`operators::streams`): its mix
/// sites' entries, and the widening of the embedded rows (`repeat`).
#[derive(Clone, Debug)]
pub struct StreamsKernels {
    pub mixing: MixingKernels,
    pub repeat: NativeKernel<magnitude_kernels::repeat_rows::Entry>,
}

/// The distinct streams entries of a program, each prepared once for every
/// site that binds it.
#[derive(Debug, Default)]
pub struct StreamsEntryKernels {
    pub(super) down:
        HashMap<MixDownBinding, NativeKernel<magnitude_kernels::stream_mix_down::Entry>>,
    pub(super) up: HashMap<MixUpBinding, NativeKernel<magnitude_kernels::stream_mix_up::Entry>>,
    pub(super) sinkhorn: HashMap<SinkhornBinding, SinkhornKernels>,
    pub(super) repeat: Option<NativeKernel<magnitude_kernels::repeat_rows::Entry>>,
}

impl StreamsEntryKernels {
    /// The entries of `binding`'s sites; none when any was not prepared.
    pub(super) fn kernels(&self, binding: BlockStreams) -> Option<StreamsKernels> {
        Some(StreamsKernels {
            mixing: match binding {
                BlockStreams::LowRank(binding) => MixingKernels::LowRank(
                    binding
                        .entries()
                        .map(|site| {
                            Some(MixSiteKernels {
                                down: self.down.get(&site.down)?.clone(),
                                up: self.up.get(&site.up)?.clone(),
                            })
                        })
                        .flatten()?,
                ),
                BlockStreams::Sinkhorn(binding) => MixingKernels::Sinkhorn(self.sinkhorn.get(&binding)?.clone()),
            },
            repeat: self.repeat.clone()?,
        })
    }
}

/// A dense branch beside a general routed branch: the dense branch's
/// expansion and its down projection into F32, the routed branch summed onto
/// zeros, and `moe_tail`, which joins both through the sublayer's post-norm
/// tail.
#[derive(Clone, Debug)]
pub struct ParallelKernels {
    pub expand: NativeKernel<dense_expand::Entry>,
    pub down: NativeKernel<magnitude_kernels::project_rows::Entry>,
    pub routed: GeneralRoutedKernels,
    pub tail: NativeKernel<magnitude_kernels::moe_tail::Entry>,
}

/// The per-layer entry (Gemma PLE), one graph after the embedding and its
/// media overlays:
/// - the embedded rows rounded to activations (`import_dense` F32 → A) and
///   projected to every layer's channels (`project_rows` into F32);
/// - the batch rows' uploaded host-table rows converted to the table's
///   resident representation (`table`);
/// - their combination (`per_layer_inputs`), copied into the program's
///   per-layer rows every block reads (`conditioning_overlay`, an F32 row
///   copy).
#[derive(Clone, Debug)]
pub struct PerLayerEntryKernels {
    pub round: NativeKernel<magnitude_kernels::import_dense::Entry>,
    pub project: NativeKernel<magnitude_kernels::project_rows::Entry>,
    pub table: TableConversion,
    pub inputs: NativeKernel<magnitude_kernels::per_layer_inputs::Entry>,
    pub copy: NativeKernel<magnitude_kernels::conditioning_overlay::Entry>,
}

/// The conversion of uploaded host-table rows into their resident form.
#[derive(Clone, Debug)]
pub enum TableConversion {
    Dense(NativeKernel<magnitude_kernels::import_dense::Entry>),
    Repack(NativeKernel<magnitude_kernels::repack_weight::Entry>),
}

/// A per-layer input sublayer: the gate over the layer's slice of the
/// per-layer inputs, and the post-norm tail of its projection.
#[derive(Clone, Debug)]
pub struct PerLayerKernels {
    pub gate: NativeKernel<magnitude_kernels::per_layer_gate::Entry>,
    pub output: PostNormKernels,
}

/// A separate draft's target taps: the tap of a tapped block's (or the
/// readout's exit) residual rows, and the readout's fusion of the taps into
/// the draft's conditioning features.
#[derive(Clone, Debug)]
pub struct TapKernels {
    pub tap: NativeKernel<tap_rows::Entry>,
    pub fusion: NativeKernel<project_rows::Entry>,
    pub features: NativeKernel<feature_rows::Entry>,
}

/// The target readout: final-norm features, and the vocabulary projection
/// that normalizes its own rows, by the plan's head placement.
#[derive(Clone, Debug)]
pub struct ReadoutKernels {
    pub features: NativeKernel<readout_features_rows::Entry>,
    pub head: ReadoutHeadKernels,
}

/// The vocabulary projection's entries (`ReadoutHead`).
#[derive(Clone, Debug)]
pub enum ReadoutHeadKernels {
    /// The head GEMV, and its projection of selected vocabulary rows.
    Packed {
        head: NativeKernel<readout_head_rows::Entry>,
        selected: NativeKernel<readout_selected_rows::Entry>,
    },
    Qualified {
        packed: Box<ReadoutHeadKernels>,
        progressive: ProgressiveReadoutKernels,
    },
}

/// A progressive head's certified levels and its full exact pass.
#[derive(Clone, Debug)]
pub struct ProgressiveReadoutKernels {
    pub top: NativeKernel<readout_top_rows::Entry>,
    pub refine: NativeKernel<readout_refine_rows::Entry>,
    pub exact: NativeKernel<readout_exact_rows::Entry>,
    pub planes: NativeKernel<readout_planes_rows::Entry>,
}

#[derive(Clone, Debug)]
pub struct RoutedKernels {
    /// Every row class of the expand form; the grouped row classes of the
    /// shared-route form.
    pub route: NativeKernel<routed_route::Entry>,
    /// Decode form (row classes up to the GEMV bound).
    pub decode: RoutedDecodeKernels,
    pub output: NativeKernel<routed_output::Entry>,
    /// Grouped form (larger row classes).
    pub group: NativeKernel<routed_group::Entry>,
    pub experts: NativeKernel<routed_experts::Entry>,
    pub combine: NativeKernel<routed_combine::Entry>,
}

/// The decode rows' routing and expansions, by the backend's
/// `DecodeForm`.
#[derive(Clone, Debug)]
pub enum RoutedDecodeKernels {
    /// `routed_route`, then the choices' and the shared expert's gate/up.
    Expand(NativeKernel<routed_expand::Entry>),
    /// The routing with the shared expert's gate/up, then the choices'.
    SharedRoute {
        route: NativeKernel<routed_route_shared::Entry>,
        choices: NativeKernel<routed_gate_up::Entry>,
    },
}

/// An attention block: normed query/gate/key/value projection, the fused
/// attention entry over the block's history codec, output projection plus
/// residual.
#[derive(Clone, Debug)]
pub struct AttentionKernels {
    pub codec: magnitude_state::CodecSpec,
    pub project: NativeKernel<attention_project::Entry>,
    pub history: AttentionHistoryKernels,
    pub output: SublayerOutput<NativeKernel<attention_output::Entry>>,
    /// A grouped output's first product (`AttentionBinding::grouped`).
    pub grouped: Option<NativeKernel<project_rows::Entry>>,
    /// A selected read's indexer and selected prefill
    /// (`AttentionBinding::selection`).
    pub selection: Option<SelectionKernels>,
    /// A low-rank query's up projection
    /// (`AttentionBinding::low_rank_query`).
    pub low_rank_query: Option<LowRankQueryKernels>,
    /// An indexing index-list source's rows and indexer
    /// (`AttentionBinding::picked`, `PickedBinding::Own` or `Reindex`).
    pub picked: Option<PickedKernels>,
}

/// An indexing index-list source's entries (`operators::attention::pooled`):
/// an own source's rows' (none for a Reindex source), and its indexer's.
#[derive(Clone, Debug)]
pub struct PickedKernels {
    pub rows: Option<PooledKernels>,
    pub indexer: IndexerKernels,
}

/// An own source's rows' entries: the tokens' `u` and `g` projections
/// (`attention_project`'s query and gate segments) in F32 (published so
/// when softmax-gated, else widened), its pooling, and its index keys'
/// projection and write.
#[derive(Clone, Debug)]
pub struct PooledKernels {
    pub project: NativeKernel<attention_project::Entry>,
    pub widen: NativeKernel<widen_rows::Entry>,
    pub pool: PoolKernel,
    pub key_project: NativeKernel<attention_project::Entry>,
    pub keys: NativeKernel<indexer_keys::Entry>,
}

/// An own source's pooling: `pool_rows` over its groups and their Window
/// domain, or at rate 1, whose tokens are their own rows and which has no
/// Window, `attention_append_dense` storing each as a window history stores
/// its keys.
#[derive(Clone, Debug)]
pub enum PoolKernel {
    Grouped(NativeKernel<pool_rows::Entry>),
    Identity(NativeKernel<attention_append_dense::Entry>),
}

/// An indexer's entries (`operators::attention::selection`): its query and
/// (projected) head-weight projections, scores and picks, and a producing
/// indexer's candidate pool's block scores and rows.
#[derive(Clone, Debug)]
pub struct IndexerKernels {
    pub query_project: NativeKernel<attention_project::Entry>,
    pub weights_project: Option<NativeKernel<attention_project::Entry>>,
    pub score: NativeKernel<indexer_score::Entry>,
    pub topk: NativeKernel<rows_topk::Entry>,
    pub candidates: Option<CandidateKernels>,
}

/// A candidate pool's entries.
#[derive(Clone, Debug)]
pub struct CandidateKernels {
    pub blocks: NativeKernel<candidate_blocks::Entry>,
    pub rows: NativeKernel<candidate_rows::Entry>,
}

/// A low-rank query's entries after the attention projection: its `qr`
/// rows widened to F32 (`widen_rows`), then normalized and projected to the
/// query heads (`attention_project`, a query segment alone).
#[derive(Clone, Debug)]
pub struct LowRankQueryKernels {
    pub widen: NativeKernel<widen_rows::Entry>,
    pub project: NativeKernel<attention_project::Entry>,
}

/// A selected read's entries: the indexer's normed query and raw-key
/// projection (`attention_project`: query rows `heads · width`, key rows
/// `width`), `indexer_select`, and the codec's prefill in its WorkList form
/// (`B` words) for the classes that score.
#[derive(Clone, Debug)]
pub struct SelectionKernels {
    pub project: NativeKernel<attention_project::Entry>,
    pub select: NativeKernel<indexer_select::Entry>,
    pub prefill: SelectedPrefillKernel,
}

/// The WorkList prefill of a selected read's history codec.
#[derive(Clone, Debug)]
pub enum SelectedPrefillKernel {
    Dense(NativeKernel<attention_prefill::Entry>),
    AffineK8V4(NativeKernel<attention_prefill_k8v4::Entry>),
}

/// A sublayer's output entries by its `SublayerTail`: the operator's own
/// output projection plus residual (`K`), or the F32 projection and the
/// post-norm row op.
#[derive(Clone, Debug)]
pub enum SublayerOutput<K> {
    Residual(K),
    PostNorm(PostNormKernels),
}

#[derive(Clone, Debug)]
pub struct PostNormKernels {
    pub project: NativeKernel<project_rows::Entry>,
    pub residual: NativeKernel<post_norm_residual::Entry>,
}

/// The fused attention entries of one history codec: `decode` for decode row
/// classes, `prefill` for the rest. The codec fixes the history planes the
/// entries take, in the order of the codec's plane descriptors.
#[derive(Clone, Debug)]
pub enum AttentionHistoryKernels {
    Dense {
        append: NativeKernel<attention_append_dense::Entry>,
        decode: DenseDecodeKernels,
        prefill: NativeKernel<attention_prefill::Entry>,
    },
    AffineK8V4 {
        append: NativeKernel<attention_append_k8v4::Entry>,
        decode: NativeKernel<attention_decode_k8v4::Entry>,
        verify: Option<NativeKernel<attention_decode_k8v4::Entry>>,
        verify_four: Option<NativeKernel<attention_decode_k8v4::Entry>>,
        verify_eight: Option<NativeKernel<attention_decode_k8v4::Entry>>,
        /// The prefill of a launch that lists no history row tiles: it reads
        /// the history in place.
        prefill: NativeKernel<attention_prefill_k8v4::Entry>,
        /// The prefill of a launch that lists the history row tiles its rows
        /// see, where the entry's forms differ by it
        /// (`StateResourcePlan::lists_history_tiles`).
        prefill_listed: Option<NativeKernel<attention_prefill_k8v4::Entry>>,
    },
}

/// What attends a dense history's decode row classes
/// (`graph::dense_decodes`).
#[derive(Clone, Debug)]
pub enum DenseDecodeKernels {
    Decode {
        single: NativeKernel<attention_decode::Entry>,
        /// The multi-row decode (draft blocks, verification), tuned on its
        /// own row classes so it can read each history tile once for all
        /// rows.
        verify: Option<NativeKernel<attention_decode::Entry>>,
    },
    /// The prefill entry tuned on the decode row classes, where the decode
    /// entry holds no form of the shape.
    Prefill(NativeKernel<attention_prefill::Entry>),
}

#[derive(Clone, Debug)]
pub struct DenseKernels {
    pub expand: NativeKernel<dense_expand::Entry>,
    pub output: SublayerOutput<NativeKernel<dense_output::Entry>>,
}

/// A recurrent block: normed projection, the state advance publishing the
/// gated rows (row-sequential `step` for small row classes, chunked for the
/// rest), and the plain residual output projection.
#[derive(Clone, Debug)]
pub struct RecurrentKernels {
    /// The chunked row classes' projection, and every row class's in the
    /// step form.
    pub project: NativeKernel<gated_delta_project::Entry>,
    /// The step row classes, by the backend's `StepForm`.
    pub step: RecurrentStepKernels,
    pub chunk: NativeKernel<gated_delta_chunk::Entry>,
    pub output: NativeKernel<attention_output::Entry>,
}

/// The step row classes' entries, by the backend's `StepForm`.
#[derive(Clone, Debug)]
pub enum RecurrentStepKernels {
    /// `gated_delta_project`, then the step that convolves.
    Step(NativeKernel<gated_delta_step::Entry>),
    /// The projection that also convolves, then the step over its channels.
    Convolved {
        project: NativeKernel<gated_delta_project_convolved::Entry>,
        step: NativeKernel<gated_delta_step_convolved::Entry>,
    },
}

/// A general routed feed-forward (`operators::routed`).
#[derive(Clone, Debug)]
pub struct GeneralRoutedKernels {
    pub select: NativeKernel<routed_select::Entry>,
    pub experts: ExpertKernels,
    /// Decode rows: the weighted down projections onto the base.
    pub down: NativeKernel<routed_down::Entry>,
    /// Grouped rows: tiles, and the unpermuted weighted sum onto the base.
    pub group: NativeKernel<routed_group::Entry>,
    pub scatter: NativeKernel<routed_scatter::Entry>,
    /// The shared expert: expansion, then its down projection onto the
    /// residual.
    pub shared: Option<(DenseExpansionKernel, NativeKernel<dense_output::Entry>)>,
    /// A latent operator's down projection (into activations) and up
    /// projection onto the base.
    pub latent: Option<(
        NativeKernel<project_rows::Entry>,
        NativeKernel<dense_output::Entry>,
    )>,
}

/// The routed experts' expansion entries: decode, then grouped.
#[derive(Clone, Debug)]
pub enum ExpertKernels {
    Gated {
        decode: NativeKernel<routed_gate_up::Entry>,
        grouped: NativeKernel<routed_experts::Entry>,
    },
    Plain {
        decode: NativeKernel<routed_up::Entry>,
        grouped: NativeKernel<routed_experts_up::Entry>,
    },
}

/// A dense expansion: gated (`dense_expand`) or up-only (`dense_up`).
#[derive(Clone, Debug)]
pub enum DenseExpansionKernel {
    Gated(NativeKernel<dense_expand::Entry>),
    Plain(NativeKernel<dense_up::Entry>),
}

/// A state-space block (`operators::state_space`): the normed projection
/// row, the state advance (`step` for small row classes, chunked for the
/// rest), the gated group norm, and the output projection plus residual.
#[derive(Clone, Debug)]
pub struct StateSpaceKernels {
    pub project: NativeKernel<attention_project::Entry>,
    pub step: NativeKernel<state_space_step::Entry>,
    pub chunk: NativeKernel<state_space_chunk::Entry>,
    pub gate: NativeKernel<state_space_gate::Entry>,
    pub output: NativeKernel<attention_output::Entry>,
}

/// A short-convolution block (`operators::short_conv`): the normed
/// `u | C` projection, the gated convolution with its window publication,
/// and the output projection plus residual.
#[derive(Clone, Debug)]
pub struct ShortConvKernels {
    pub project: NativeKernel<short_conv_project::Entry>,
    pub rows: NativeKernel<short_conv_rows::Entry>,
    pub output: SublayerOutput<NativeKernel<attention_output::Entry>>,
}
