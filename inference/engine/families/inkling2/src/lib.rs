//! Inkling 2 (`inkling2`) artifact recognition and numerical interpretation
//! (Thinking Machines Lab's Inkling 2 Small and looped Inkling 2 Mini /
//! Varloop).
//!
//! Reads the standard `llama.cpp` `inkling2` GGUF schema (metadata keys and
//! unfused tensor names, including PR #32695 loop extensions) in a
//! format-agnostic way: NVFP4 + companion F32 `.scale`, MXFP8 (`F8_E4M3` +
//! `I8` UE8M0 `.scale`), BF16, and community quantizations all bind through
//! [`Tensors`] with no format-specific logic.
//!
//! Looped models (`num_loops > 1`) are unrolled into `Decoder.blocks` (one
//! logical block per layer visit) reusing identical [`WeightDescriptor`]s from
//! each physical layer so resident weights and streamed expert banks never
//! duplicate storage.

mod family;
pub mod projector;

pub use family::Inkling2Family;

use magnitude_artifacts::{
    PackageIdentity,
    gguf::{Directory, Value},
};
use magnitude_family_common::{Metadata, Tensors, binders};
use magnitude_family_contracts::{
    ActivationDType, ActivationFunction, Attention, AttentionGate, AttentionSinks, Block,
    BlockAttentionInput, BlockAttentionNorm, BlockAttentionQueries, BlockTransition,
    CausalShortConv, CrossPassInput, CrossPassMessage, Decoder, DynamicConvTap, EmbeddingScale,
    EntryForm, EntryNorm, ExitForm, ExitNorm, ExpertSelection, FamilyError, FamilyId,
    FeedForwardUp, GateFunction, GateGranularity, GluLimit, HeadNorm, HistoryRows, HistorySource,
    ImportTransform, InputNorm, InputSemantics, LogScale, MediaRowAttention, MessageController,
    ModelDefinition, Operator, OutputForm, OutputProjection, ProjectedRows, QueryKeys,
    QueryProjection, RelativeBias, ResidualForm, RmsNorm, Rotary, RouteNormalization, RoutedFfn,
    Router, RouterInput, RowProducer, RowRounding, ScoreFunction, Sublayer, TokenDomain,
    UnweightedRms, ValueNorm, ValueSource, WeightDescriptor, checked_product,
};

/// The primary GGUF architecture and family identity.
pub const ARCHITECTURE: &str = "inkling2";

/// The mainline `llama.cpp` GGUF architecture for looped Inkling 2 models
/// (`block_count` is physical layers; loop tensors live per core block).
pub const ARCHITECTURE_LOOP: &str = "inkling2_loop";

/// Fixed L2-clipping denominator epsilon (`||x||_2 + 1e-6`), matching
/// llama.cpp's `clip()` and SGLang's `l2_clip`.
const CLIP_EPSILON: f64 = 1e-6;

/// Recognize an `inkling2` or `inkling2_loop` artifact from
/// `general.architecture`, returning the matched architecture prefix.
pub fn recognize(directory: &Directory) -> Result<&'static str, FamilyError> {
    match directory
        .value("general.architecture")
        .and_then(Value::string)
    {
        Some(ARCHITECTURE) => Ok(ARCHITECTURE),
        Some(ARCHITECTURE_LOOP) => Ok(ARCHITECTURE_LOOP),
        architecture => Err(FamilyError::invalid(format!(
            "GGUF architecture {architecture:?} is neither {ARCHITECTURE:?} nor {ARCHITECTURE_LOOP:?}"
        ))),
    }
}

fn product(dimensions: &[u64]) -> Result<u64, FamilyError> {
    checked_product(dimensions).map_err(|error| FamilyError::invalid(error.to_string()))
}

/// Extract row `row` of a 2D `[rows, width]` descriptor as a `[width]` vector
/// descriptor (preserving any prior import transforms).
fn slice_vector(weight: &WeightDescriptor, row: u64) -> Result<WeightDescriptor, FamilyError> {
    let mut sliced = weight
        .slice_rows(row, 1)
        .map_err(|error| FamilyError::invalid(error.to_string()))?;
    let width = sliced
        .shape
        .last()
        .copied()
        .ok_or_else(|| FamilyError::invalid("cannot flatten scalar slice"))?;
    sliced.transforms.push(ImportTransform::Flatten);
    sliced.shape = vec![width];
    Ok(sliced)
}

/// Loop geometry when `inkling2.loop.core_end` is present.
///
/// Validated at construction so `extra_visits`, `core_end + extra_visits`, and
/// all visit-index arithmetic are guaranteed not to overflow and to stay within
/// `block_count`.
#[derive(Clone, Copy, Debug)]
struct LoopLayout {
    core_start: u64,
    core_end: u64,
    num_passes: u64,
    gate_rank: u64,
    is_varloop: bool,
    extra_visits: u64,
}

impl LoopLayout {
    #[allow(clippy::too_many_arguments)]
    fn new(
        core_start: u64,
        core_end: u64,
        num_passes: u64,
        gate_rank: u64,
        is_varloop: bool,
        attn_res_block_size: u64,
        block_count: u64,
        core_end_key: &str,
        num_loops_key: &str,
    ) -> Result<Self, FamilyError> {
        let min_passes = if is_varloop { 1 } else { 2 };
        if core_start == 0
            || core_start >= core_end
            || num_passes < min_passes
            || !core_start.is_multiple_of(attn_res_block_size)
            || !(core_end - core_start).is_multiple_of(attn_res_block_size)
        {
            return Err(FamilyError::metadata(
                core_end_key,
                format!(
                    "invalid loop layout: core [{core_start}, {core_end}) x {num_passes} passes \
                     (block_size={attn_res_block_size})"
                ),
            ));
        }
        let n_core = core_end - core_start;
        let extra_visits = (num_passes - 1)
            .checked_mul(n_core)
            .ok_or_else(|| {
                FamilyError::metadata(
                    num_loops_key,
                    format!(
                        "loop core [{core_start}, {core_end}) x {num_passes} overflows visit count"
                    ),
                )
            })?;
        let min_blocks = core_end.checked_add(extra_visits).ok_or_else(|| {
            FamilyError::metadata(
                num_loops_key,
                format!("loop core [{core_start}, {core_end}) x {num_passes} overflows block_count"),
            )
        })?;
        if min_blocks > block_count {
            return Err(FamilyError::metadata(
                num_loops_key,
                format!(
                    "loop core [{core_start}, {core_end}) x {num_passes} exceeds block_count {block_count}"
                ),
            ));
        }
        Ok(Self {
            core_start,
            core_end,
            num_passes,
            gate_rank,
            is_varloop,
            extra_visits,
        })
    }

    fn n_core(self) -> u64 {
        self.core_end - self.core_start
    }

    fn extra_visits(self) -> u64 {
        self.extra_visits
    }

    /// Map a logical visit `il` to `(physical_layer, pass)`.
    fn visit(self, il: u64) -> (u64, u64) {
        if self.num_passes < 2 || il < self.core_start {
            return (il, 0);
        }
        let n_core = self.n_core();
        let core_visits = self.extra_visits + n_core;
        if il < self.core_start + core_visits {
            let offset = il - self.core_start;
            (self.core_start + offset % n_core, offset / n_core)
        } else {
            (il - self.extra_visits, 0)
        }
    }

    /// First logical visit of physical layer `ip`.
    fn first_visit(self, ip: u64) -> u64 {
        if ip < self.core_end {
            ip
        } else {
            ip + self.extra_visits
        }
    }

    fn in_core(self, ip: u64) -> bool {
        ip >= self.core_start && ip < self.core_end
    }
}

/// All bound weights and operators for one physical layer `ip`.
/// Cloned across logical visits when the layer belongs to a looped core.
struct PhysicalLayer {
    attn_queries: BlockAttentionQueries,
    attn_norm: BlockAttentionNorm,
    attention: Box<Attention>,
    attn_post_conv: CausalShortConv,
    ffn_query: WeightDescriptor,
    ffn_norm: RmsNorm,
    routed: Box<RoutedFfn>,
    ffn_post_conv: CausalShortConv,
    /// `[attn, k, v, mlp]` controllers for core layers on looped models.
    controllers: Option<Box<[MessageController; 4]>>,
    /// Per-layer `[attn, k, v, mlp]` cross-pass message queries when stored on
    /// the core block (`inkling2_loop`: `blk.{ip}.loop_msg_query.weight`).
    loop_msg_queries: Option<Box<[WeightDescriptor; 4]>>,
}

struct LayerSpec {
    hidden: u64,
    epsilon: f64,
    qk_clip: f64,
    conv_k: u64,
    d_rel: u64,
    rel_extent: u64,
    rel_clip: f64,
    global_log_scale: Option<LogScale>,
    head_dim_full: u64,
    head_dim_swa: u64,
    sliding_window: u64,
    experts: u64,
    expert_weights_scale: f64,
    situ_limit: GluLimit,
    static_conv_scale: f64,
    dynamic_conv_scale: f64,
    loop_layout: Option<LoopLayout>,
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn bind_physical_layer(
    tensors: &mut Tensors<'_>,
    spec: &LayerSpec,
    ip: u64,
    swa: bool,
    n_head: u64,
    n_head_kv: u64,
    n_ff_exp: u64,
    n_exp_used: u64,
) -> Result<Box<PhysicalLayer>, FamilyError> {
    let LayerSpec {
        hidden,
        epsilon,
        qk_clip,
        conv_k,
        d_rel,
        rel_extent,
        rel_clip,
        global_log_scale,
        head_dim_full,
        head_dim_swa,
        sliding_window,
        experts,
        expert_weights_scale,
        situ_limit,
        static_conv_scale,
        dynamic_conv_scale,
        loop_layout,
    } = *spec;
    let rms = |weight: WeightDescriptor| RmsNorm { weight, epsilon };
    let head_dim = if swa { head_dim_swa } else { head_dim_full };
    if n_head == 0
        || n_head_kv == 0
        || !n_head.is_multiple_of(n_head_kv)
        || n_exp_used == 0
        || n_exp_used > experts
        || n_ff_exp == 0
    {
        return Err(FamilyError::invalid(format!(
            "inkling2: invalid attention or expert dimensions at physical layer {ip}"
        )));
    }

    let query_rows = product(&[n_head, head_dim])?;
    let kv_rows = product(&[n_head_kv, head_dim])?;
    let rel_rows = product(&[n_head, d_rel])?;
    let extent = if swa { sliding_window } else { rel_extent };
    let p = format!("blk.{ip}.");

    let attn_queries = if ip == 0 {
        BlockAttentionQueries::Partial
    } else if swa {
        BlockAttentionQueries::Single(
            tensors.bind(&format!("{p}attn_res_query.weight"), &[hidden])?,
        )
    } else {
        let qkv = tensors.bind(&format!("{p}attn_res_query.weight"), &[3, hidden])?;
        BlockAttentionQueries::SplitQkv {
            query: slice_vector(&qkv, 0)?,
            key: slice_vector(&qkv, 1)?,
            value: slice_vector(&qkv, 2)?,
        }
    };

    let attn_norm = if swa {
        BlockAttentionNorm::Shared(rms(
            tensors.bind(&format!("{p}attn_norm.weight"), &[hidden])?,
        ))
    } else {
        BlockAttentionNorm::SplitQkv {
            query: rms(tensors.bind(&format!("{p}attn_q_inp_norm.weight"), &[hidden])?),
            key: rms(tensors.bind(&format!("{p}attn_k_inp_norm.weight"), &[hidden])?),
            value: rms(tensors.bind(&format!("{p}attn_v_inp_norm.weight"), &[hidden])?),
        }
    };

    let wq = tensors.bind(&format!("{p}attn_q.weight"), &[query_rows, hidden])?;
    let wk = tensors.bind(&format!("{p}attn_k.weight"), &[kv_rows, hidden])?;
    let wv = tensors.bind(&format!("{p}attn_v.weight"), &[kv_rows, hidden])?;
    let wo = tensors.bind(&format!("{p}attn_output.weight"), &[hidden, query_rows])?;
    let gate = tensors.bind(&format!("{p}attn_gate.weight"), &[query_rows, hidden])?;
    let q_clip = tensors.bind(&format!("{p}attn_q_clip.weight"), &[n_head, head_dim])?;
    let sink_proj = tensors.bind(&format!("{p}attn_sink_proj.weight"), &[n_head, hidden])?;

    let rel_q = tensors.bind(&format!("{p}attn_rel.weight"), &[rel_rows, hidden])?;
    let rel_bias = tensors.bind(&format!("{p}attn_rel_bias.weight"), &[extent, d_rel])?;
    let rel_clip_w = tensors.bind(&format!("{p}attn_rel_clip.weight"), &[n_head, d_rel])?;

    let (k_dyn, v_dyn, kv_conv_scale) = if swa {
        (None, None, static_conv_scale)
    } else {
        let dyn_rows = product(&[n_head_kv, conv_k])?;
        let k_proj = tensors.bind(&format!("{p}attn_k_conv_proj.weight"), &[dyn_rows, hidden])?;
        let v_proj = tensors.bind(&format!("{p}attn_v_conv_proj.weight"), &[dyn_rows, hidden])?;
        (
            Some(DynamicConvTap {
                proj: k_proj,
                heads: n_head_kv,
            }),
            Some(DynamicConvTap {
                proj: v_proj,
                heads: n_head_kv,
            }),
            dynamic_conv_scale,
        )
    };
    let key_conv = CausalShortConv {
        kernel_size: conv_k,
        weight: tensors.bind(&format!("{p}attn_k_conv.weight"), &[kv_rows, conv_k])?,
        dynamic: k_dyn,
        residual_scale: kv_conv_scale,
    };
    let value_conv = CausalShortConv {
        kernel_size: conv_k,
        weight: tensors.bind(&format!("{p}attn_v_conv.weight"), &[kv_rows, conv_k])?,
        dynamic: v_dyn,
        residual_scale: kv_conv_scale,
    };
    let attn_post_conv = CausalShortConv {
        kernel_size: conv_k,
        weight: tensors.bind(&format!("{p}attn_post_conv.weight"), &[hidden, conv_k])?,
        dynamic: None,
        residual_scale: static_conv_scale,
    };
    let ffn_post_conv = CausalShortConv {
        kernel_size: conv_k,
        weight: tensors.bind(&format!("{p}ffn_post_conv.weight"), &[hidden, conv_k])?,
        dynamic: None,
        residual_scale: static_conv_scale,
    };

    let attention = Box::new(Attention {
        heads: n_head,
        kv_heads: n_head_kv,
        width: head_dim,
        query: QueryProjection::Full(wq),
        query_keys: QueryKeys::Direct,
        gate: AttentionGate::Separate {
            weight: gate,
            function: GateFunction::Sigmoid,
            granularity: GateGranularity::Element,
        },
        query_norm: HeadNorm::L2Clip {
            threshold: qk_clip,
            epsilon: CLIP_EPSILON,
            scale: Some(q_clip),
        },
        history: vec![HistorySource::visible(HistoryRows::Own(
            RowProducer::Projected(ProjectedRows {
                key: wk,
                value: ValueSource::Projected(wv),
                key_conv: Some(key_conv),
                value_conv: Some(value_conv),
                key_norm: HeadNorm::L2Clip {
                    threshold: qk_clip,
                    epsilon: CLIP_EPSILON,
                    scale: None,
                },
                value_norm: ValueNorm::RmsUnweighted(UnweightedRms { epsilon }),
                domain: if swa {
                    TokenDomain::Window {
                        tokens: sliding_window,
                    }
                } else {
                    TokenDomain::Token
                },
                rounding: RowRounding::None,
            }),
        ))],
        rotary: Rotary::None,
        relative_bias: Some(RelativeBias {
            d_rel,
            extent,
            query_proj: rel_q,
            rel_clip: HeadNorm::L2Clip {
                threshold: rel_clip,
                epsilon: CLIP_EPSILON,
                scale: Some(rel_clip_w),
            },
            bias_proj: rel_bias,
        }),
        log_scale: if swa { None } else { global_log_scale },
        scale: 1.0 / (head_dim as f64).sqrt(),
        media_rows: MediaRowAttention::Causal,
        sinks: AttentionSinks::Dynamic(sink_proj),
        output: OutputProjection::Full(wo),
    });

    let ffn_query = tensors.bind(&format!("{p}ffn_res_query.weight"), &[hidden])?;
    let ffn_norm = rms(tensors.bind(&format!("{p}ffn_norm.weight"), &[hidden])?);
    let router_w = tensors.bind(&format!("{p}ffn_gate_inp.weight"), &[experts, hidden])?;
    let bias_name = format!("{p}exp_probs_b.bias");
    let router_bias = if tensors.contains(&bias_name) {
        Some(tensors.bind(&bias_name, &[experts])?)
    } else {
        None
    };
    let gate_up = tensors.bind(
        &format!("{p}ffn_gate_up_exps.weight"),
        &[experts, product(&[2, n_ff_exp])?, hidden],
    )?;
    let expert_gate = gate_up
        .slice_rows(0, n_ff_exp)
        .map_err(|error| FamilyError::invalid(error.to_string()))?;
    let expert_up = gate_up
        .slice_rows(n_ff_exp, n_ff_exp)
        .map_err(|error| FamilyError::invalid(error.to_string()))?;
    let expert_down = tensors.bind(
        &format!("{p}ffn_down_exps.weight"),
        &[experts, hidden, n_ff_exp],
    )?;

    let routed = Box::new(RoutedFfn {
        experts,
        selected: n_exp_used,
        intermediate: n_ff_exp,
        router: Router {
            weight: router_w,
            input: RouterInput::Operator,
            score: ScoreFunction::Sigmoid,
            selection: ExpertSelection::TopK { bias: router_bias },
            media_bias: None,
            normalization: RouteNormalization::SumPlusEpsilon(binders::ROUTE_EPSILON),
            scale: expert_weights_scale,
        },
        expert_up: FeedForwardUp::Gated {
            activation: ActivationFunction::Silu,
            limit: Some(situ_limit),
            gate: expert_gate,
            up: expert_up,
        },
        expert_down,
        expert_scale: None,
        latent: None,
        shared: None,
    });

    let (controllers, loop_msg_queries) = match loop_layout {
        Some(layout) if layout.in_core(ip) => {
            let rank = layout.gate_rank;
            let two_hidden = product(&[2, hidden])?;
            let per_block_schema = tensors.contains(&format!("{p}loop_q_ctrl_down.weight"));
            let modules = if per_block_schema {
                ["q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl"]
            } else {
                ["attn", "k", "v", "mlp"]
            };
            let msg_q_name = format!("{p}loop_msg_query.weight");
            if layout.num_passes > 1 {
                let mut ctrls = Vec::with_capacity(4);
                for module in modules {
                    ctrls.push(MessageController {
                        rank,
                        down: tensors.bind(
                            &format!("{p}loop_{module}_down.weight"),
                            &[rank, two_hidden],
                        )?,
                        gain: tensors.bind(
                            &format!("{p}loop_{module}_gain.weight"),
                            &[hidden, rank],
                        )?,
                        forget: tensors.bind(
                            &format!("{p}loop_{module}_forget.weight"),
                            &[hidden, rank],
                        )?,
                    });
                }
                let arr: [MessageController; 4] = ctrls.try_into().expect("4 controllers");
                let msg_q = if per_block_schema {
                    let q4 = tensors.bind(&msg_q_name, &[4, hidden])?;
                    Some(Box::new([
                        slice_vector(&q4, 0)?,
                        slice_vector(&q4, 1)?,
                        slice_vector(&q4, 2)?,
                        slice_vector(&q4, 3)?,
                    ]))
                } else {
                    None
                };
                (Some(Box::new(arr)), msg_q)
            } else {
                for module in modules {
                    tensors.bind_unused(
                        &format!("{p}loop_{module}_down.weight"),
                        &[rank, two_hidden],
                    )?;
                    tensors.bind_unused(
                        &format!("{p}loop_{module}_gain.weight"),
                        &[hidden, rank],
                    )?;
                    tensors.bind_unused(
                        &format!("{p}loop_{module}_forget.weight"),
                        &[hidden, rank],
                    )?;
                }
                if per_block_schema {
                    tensors.bind_unused(&msg_q_name, &[4, hidden])?;
                }
                (None, None)
            }
        }
        _ => (None, None),
    };

    Ok(Box::new(PhysicalLayer {
        attn_queries,
        attn_norm,
        attention,
        attn_post_conv,
        ffn_query,
        ffn_norm,
        routed,
        ffn_post_conv,
        controllers,
        loop_msg_queries,
    }))
}

#[inline(never)]
fn unroll_visit_block(
    phys: &PhysicalLayer,
    ip: u64,
    pass: u64,
    attn_res_block_size: u64,
    loop_layout: Option<LoopLayout>,
    loop_queries: Option<&(WeightDescriptor, WeightDescriptor)>,
) -> Result<Block, FamilyError> {
    let transition = match loop_layout {
        Some(layout) if layout.num_passes > 1 && ip == layout.core_start => {
            if pass == 0 {
                Some(BlockTransition::SnapshotPrelude)
            } else {
                Some(BlockTransition::ResetToPreludeAndBank)
            }
        }
        _ => None,
    };

    let (attn_cross_pass, ffn_cross_pass) = if pass > 0 {
        let layout = loop_layout.expect("pass > 0 implies loop_layout");
        let ctrls = phys
            .controllers
            .as_ref()
            .expect("core layer has controllers");
        let (q_attn, q_k, q_v, q_mlp) = if let Some(msg_q) = phys.loop_msg_queries.as_ref() {
            (
                msg_q[0].clone(),
                msg_q[1].clone(),
                msg_q[2].clone(),
                msg_q[3].clone(),
            )
        } else {
            let (res_q, kv_q) = loop_queries.expect("pass > 0 implies loop_queries");
            let row = 2 * (ip - layout.core_start);
            (
                slice_vector(res_q, row)?,
                slice_vector(kv_q, row)?,
                slice_vector(kv_q, row + 1)?,
                slice_vector(res_q, row + 1)?,
            )
        };
        (
            CrossPassInput::SplitQkv {
                query: Box::new(CrossPassMessage {
                    query: q_attn,
                    controller: ctrls[0].clone(),
                }),
                key: Box::new(CrossPassMessage {
                    query: q_k,
                    controller: ctrls[1].clone(),
                }),
                value: Box::new(CrossPassMessage {
                    query: q_v,
                    controller: ctrls[2].clone(),
                }),
            },
            CrossPassInput::Single(Box::new(CrossPassMessage {
                query: q_mlp,
                controller: ctrls[3].clone(),
            })),
        )
    } else {
        (CrossPassInput::None, CrossPassInput::None)
    };

    let attn_output_form = if ip.is_multiple_of(attn_res_block_size) {
        OutputForm::BlockCommit
    } else {
        OutputForm::Residual
    };

    Ok(Block {
        transition,
        sublayers: vec![
            Sublayer {
                input: InputNorm::BlockAttention(BlockAttentionInput {
                    queries: phys.attn_queries.clone(),
                    cross_pass: attn_cross_pass,
                    norm: phys.attn_norm.clone(),
                }),
                op: Operator::Attention(phys.attention.clone()),
                post_conv: Some(phys.attn_post_conv.clone()),
                output: attn_output_form,
            },
            Sublayer {
                input: InputNorm::BlockAttention(BlockAttentionInput {
                    queries: BlockAttentionQueries::Single(phys.ffn_query.clone()),
                    cross_pass: ffn_cross_pass,
                    norm: BlockAttentionNorm::Shared(phys.ffn_norm.clone()),
                }),
                op: Operator::RoutedFfn(phys.routed.clone()),
                post_conv: Some(phys.ffn_post_conv.clone()),
                output: OutputForm::Residual,
            },
        ],
    })
}

/// Validate an optional duplicate alias key against its canonical integer value.
fn check_alias_int(m: &Metadata<'_>, alias: &str, canonical: u64) -> Result<(), FamilyError> {
    if let Some(value) = m.optional_integer(alias)? {
        if value != canonical {
            let key = m.key(alias);
            return Err(FamilyError::metadata(
                &key,
                format!("{key} ({value}) disagrees with canonical value ({canonical})"),
            ));
        }
    }
    Ok(())
}

/// Validate an optional duplicate alias key against its canonical float value.
fn check_alias_f64(m: &Metadata<'_>, alias: &str, canonical: f64) -> Result<(), FamilyError> {
    if let Some(value) = m.optional_number(alias)? {
        if (value - canonical).abs() > 1e-4 * canonical.abs().max(1.0) {
            let key = m.key(alias);
            return Err(FamilyError::metadata(
                &key,
                format!("{key} ({value}) disagrees with canonical value ({canonical})"),
            ));
        }
    }
    Ok(())
}

/// Inspect already-opened package components.
pub fn inspect_components(
    directory: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
) -> Result<ModelDefinition, FamilyError> {
    let arch = recognize(directory)?;
    let is_loop_arch = arch == ARCHITECTURE_LOOP;
    let m = Metadata::new(directory, arch);

    let declared_block_count = m.bounded_block_count("block_count")?;
    let context_limit = m.integer("context_length")?;
    let hidden = m.integer("embedding_length")?;
    let _ = m.optional_integer("feed_forward_length")?;
    if let Some(rope_dims) = m.optional_count("rope.dimension_count")? {
        if rope_dims != 0 {
            let key = m.key("rope.dimension_count");
            return Err(FamilyError::metadata(
                &key,
                format!("{key} must be 0 (inkling2 uses NoPE + relative bias)"),
            ));
        }
    }

    let epsilon = m.positive_number("attention.layer_norm_rms_epsilon")?;
    let qk_clip = m.positive_number("attention.qk_clip")?;
    let attn_res_block_size = m.integer("attn_res.block_size")?;
    let conv_k = m.integer("shortconv.l_cache")?;
    if !(2..=64).contains(&conv_k) {
        let key = m.key("shortconv.l_cache");
        return Err(FamilyError::metadata(
            &key,
            format!("{key} {conv_k} must be in 2..=64"),
        ));
    }

    let d_rel = m.integer("attention.relative_length")?;
    let rel_extent = m.integer("attention.relative_extent")?;
    let rel_clip = m.positive_number("attention.relative_clip")?;
    let log_scale_floor = m.integer("attention.log_scale_floor")?;
    let log_scale_alpha = m.number("attention.log_scale_alpha")?;
    if !log_scale_alpha.is_finite() || log_scale_alpha < 0.0 {
        let key = m.key("attention.log_scale_alpha");
        return Err(FamilyError::metadata(
            &key,
            format!("{key} must be finite and nonnegative"),
        ));
    }
    let global_log_scale = (log_scale_alpha > 0.0).then_some(LogScale {
        floor: log_scale_floor,
        alpha: log_scale_alpha,
    });

    let vocab_padded = m.integer("vocab_size")?;
    let vocab_unpadded = m.integer("vocab_size_unpadded")?;
    if vocab_unpadded > vocab_padded {
        let key = m.key("vocab_size_unpadded");
        return Err(FamilyError::metadata(
            &key,
            format!("{key} ({vocab_unpadded}) exceeds vocab_size ({vocab_padded})"),
        ));
    }

    let head_dim_full = m.integer("attention.key_length")?;
    if m.integer("attention.value_length")? != head_dim_full {
        let key = m.key("attention.value_length");
        return Err(FamilyError::metadata(
            &key,
            "attention.value_length must equal attention.key_length",
        ));
    }
    let head_dim_swa = m
        .optional_integer("attention.key_length_swa")?
        .unwrap_or(head_dim_full);
    if m.optional_integer("attention.value_length_swa")?
        .unwrap_or(head_dim_swa)
        != head_dim_swa
    {
        let key = m.key("attention.value_length_swa");
        return Err(FamilyError::metadata(
            &key,
            "attention.value_length_swa must equal attention.key_length_swa",
        ));
    }

    let sliding_window = m.integer("attention.sliding_window")?;
    let is_swa = m.flags("attention.sliding_window_pattern", declared_block_count)?;
    let heads = m.uniform_or_layer_counts("attention.head_count", declared_block_count)?;
    let kv_heads = m.uniform_or_layer_counts("attention.head_count_kv", declared_block_count)?;

    let experts = m.integer("expert_count")?;
    let expert_used = m.uniform_or_layer_counts("expert_used_count", declared_block_count)?;
    let expert_ff = m.uniform_or_layer_counts("expert_feed_forward_length", declared_block_count)?;
    let expert_weights_scale = m
        .optional_positive_number("expert_weights_scale")?
        .unwrap_or(1.0);

    let situ_beta = m.positive_number("activation.situ_beta")?;
    let situ_linear_beta = m.positive_number("activation.situ_linear_beta")?;
    let situ_limit = GluLimit::softcap(situ_beta, situ_linear_beta)
        .map_err(|error| FamilyError::invalid(error.to_string()))?;

    let softcap = m
        .optional_number("final_logit_softcapping")?
        .filter(|cap| *cap != 0.0);

    // Optional duplicate alias keys emitted by some community converters
    // (e.g. Unsloth): verify consistency with the canonical keys above.
    check_alias_int(&m, "d_rel", d_rel)?;
    check_alias_int(&m, "rel_extent", rel_extent)?;
    check_alias_int(&m, "rel_extent_swa", sliding_window)?;
    check_alias_f64(&m, "rel_clip_threshold", rel_clip)?;
    check_alias_f64(&m, "qk_clip_threshold", qk_clip)?;
    check_alias_f64(&m, "log_scaling_alpha", log_scale_alpha)?;
    check_alias_int(&m, "log_scaling_n_floor", log_scale_floor)?;
    check_alias_f64(&m, "mlp_gate_softcap", situ_beta)?;
    check_alias_f64(&m, "mlp_up_softcap", situ_linear_beta)?;
    check_alias_int(&m, "shortconv_kernel", conv_k)?;
    check_alias_int(&m, "unpadded_vocab_size", vocab_unpadded)?;
    if let Some(flag) = m.optional_flag("attn_res_normalize_values")? {
        if !flag {
            let key = m.key("attn_res_normalize_values");
            return Err(FamilyError::metadata(&key, format!("{key} must be true")));
        }
    }
    if let Some(flag) = m.optional_flag("value_norm")? {
        if !flag {
            let key = m.key("value_norm");
            return Err(FamilyError::metadata(&key, format!("{key} must be true")));
        }
    }

    // Optional loop configuration (Inkling 2 Mini / Varloop).
    // `inkling2_loop` stores physical `block_count` and `loop.{start,end,count,varying}`;
    // `inkling2` stores logical `block_count` and `loop.core_{start,end}`, `num_loops`, `loop.is_varloop`.
    let (block_count, n_phys, loop_layout) = if is_loop_arch {
        let core_start = m.integer("loop.start")?;
        let core_end = m.integer("loop.end")?;
        let num_passes = m.integer("loop.count")?;
        let gate_rank = m.integer("loop.gate_rank")?;
        let is_varloop = m.optional_flag("loop.varying")?.unwrap_or(false);
        let core_end_key = m.key("loop.end");
        let num_loops_key = m.key("loop.count");
        if core_end > declared_block_count {
            return Err(FamilyError::metadata(
                &core_end_key,
                format!("loop.end ({core_end}) exceeds physical block_count ({declared_block_count})"),
            ));
        }
        let n_core = core_end.saturating_sub(core_start);
        let extra = num_passes
            .saturating_sub(1)
            .checked_mul(n_core)
            .ok_or_else(|| {
                FamilyError::metadata(&num_loops_key, "loop pass count overflows logical blocks")
            })?;
        let logical_blocks = declared_block_count.checked_add(extra).ok_or_else(|| {
            FamilyError::metadata(&num_loops_key, "logical block count overflows u64")
        })?;
        let layout = LoopLayout::new(
            core_start,
            core_end,
            num_passes,
            gate_rank,
            is_varloop,
            attn_res_block_size,
            logical_blocks,
            &core_end_key,
            &num_loops_key,
        )?;
        (logical_blocks, declared_block_count, Some(layout))
    } else {
        let layout = match m.optional_integer("loop.core_end")? {
            None => None,
            Some(core_end) => {
                let core_start = m.integer("loop.core_start")?;
                let num_passes = m.integer("num_loops")?;
                let gate_rank = m.integer("loop.gate_rank")?;
                let is_varloop = m.optional_flag("loop.is_varloop")?.unwrap_or(false);
                let core_end_key = m.key("loop.core_end");
                let num_loops_key = m.key("num_loops");
                Some(LoopLayout::new(
                    core_start,
                    core_end,
                    num_passes,
                    gate_rank,
                    is_varloop,
                    attn_res_block_size,
                    declared_block_count,
                    &core_end_key,
                    &num_loops_key,
                )?)
            }
        };
        let n_phys = declared_block_count - layout.map_or(0, LoopLayout::extra_visits);
        (declared_block_count, n_phys, layout)
    };

    // Validate per-visit consistency for looped `inkling2` models (where metadata
    // arrays are indexed by logical visit rather than physical layer).
    if !is_loop_arch {
        if let Some(layout) = loop_layout {
            if layout.num_passes > 1 {
                for il in 0..block_count {
                    let (ip, _) = layout.visit(il);
                    let iv = layout.first_visit(ip) as usize;
                    let il = il as usize;
                    if is_swa[il] != is_swa[iv]
                        || heads[il] != heads[iv]
                        || kv_heads[il] != kv_heads[iv]
                        || expert_used[il] != expert_used[iv]
                        || expert_ff[il] != expert_ff[iv]
                    {
                        return Err(FamilyError::invalid(format!(
                            "inkling2: layer visit {il} disagrees with first visit {iv} of physical layer {ip}"
                        )));
                    }
                }
            }
        }
    }

    let rms = |weight: WeightDescriptor| RmsNorm { weight, epsilon };
    let static_conv_scale = 1.0 / ((conv_k + 1) as f64).sqrt();
    let dynamic_conv_scale = 1.0 / ((2 * conv_k + 1) as f64).sqrt();

    let mut tensors = Tensors::new(directory);
    let embedding = tensors.bind("token_embd.weight", &[vocab_padded, hidden])?;
    let embed_norm = rms(tensors.bind("token_embd_norm.weight", &[hidden])?);
    let output = tensors.bind("output.weight", &[vocab_padded, hidden])?;
    let output_norm = rms(tensors.bind("output_norm.weight", &[hidden])?);
    let output_res_query = tensors.bind("output_res_query.weight", &[hidden])?;

    let has_global_loop_queries = tensors.contains("loop_res_query.weight");
    let loop_queries = match loop_layout {
        Some(layout) if has_global_loop_queries && layout.num_passes > 1 => {
            let n_query = product(&[2, layout.n_core()])?;
            let res_q = tensors.bind("loop_res_query.weight", &[n_query, hidden])?;
            let kv_q = tensors.bind("loop_kv_res_query.weight", &[n_query, hidden])?;
            Some((res_q, kv_q))
        }
        Some(layout) if has_global_loop_queries && layout.is_varloop => {
            let n_query = product(&[2, layout.n_core()])?;
            tensors.bind_unused("loop_res_query.weight", &[n_query, hidden])?;
            tensors.bind_unused("loop_kv_res_query.weight", &[n_query, hidden])?;
            None
        }
        _ => None,
    };

    // Bind every physical layer's weights once.
    let spec = LayerSpec {
        hidden,
        epsilon,
        qk_clip,
        conv_k,
        d_rel,
        rel_extent,
        rel_clip,
        global_log_scale,
        head_dim_full,
        head_dim_swa,
        sliding_window,
        experts,
        expert_weights_scale,
        situ_limit,
        static_conv_scale,
        dynamic_conv_scale,
        loop_layout,
    };
    let mut physical = Vec::with_capacity(n_phys as usize);
    for ip in 0..n_phys {
        let iv = if is_loop_arch {
            ip as usize
        } else {
            loop_layout.map_or(ip, |l| l.first_visit(ip)) as usize
        };
        physical.push(bind_physical_layer(
            &mut tensors,
            &spec,
            ip,
            is_swa[iv],
            heads[iv],
            kv_heads[iv],
            expert_ff[iv],
            expert_used[iv],
        )?);
    }

    tensors.finish()?;

    // Unroll logical layer visits into `Decoder.blocks`.
    let mut blocks = Vec::with_capacity(block_count as usize);
    for il in 0..block_count {
        let (ip, pass) = loop_layout.map_or((il, 0), |l| l.visit(il));
        blocks.push(unroll_visit_block(
            &physical[ip as usize],
            ip,
            pass,
            attn_res_block_size,
            loop_layout,
            loop_queries.as_ref(),
        )?);
    }

    let vision = projector
        .map(|projector| projector::describe_projector(projector, hidden))
        .transpose()?;

    let definition = ModelDefinition {
        family: FamilyId(ARCHITECTURE.into()),
        artifact_identity: identity,
        inputs: InputSemantics { coordinate_axes: 1 },
        decoder: Decoder {
            activation_dtype: ActivationDType::BF16,
            hidden,
            vocabulary: vocab_padded,
            context_limit,
            residual: ResidualForm::BlockAttention {
                block_size: attn_res_block_size,
                normalize_values: true,
                norm_epsilon: epsilon,
            },
            entry: EntryForm {
                embedding,
                scale: EmbeddingScale::Unit,
                norm: Some(EntryNorm::Rms {
                    norm: embed_norm,
                    text_only: true,
                }),
                per_layer: None,
            },
            blocks,
            exit: ExitForm {
                norm: ExitNorm::BlockAttention {
                    query: output_res_query,
                    norm: output_norm,
                },
                output,
                softcap,
                unpadded_vocabulary: Some(vocab_unpadded),
            },
        },
        head: None,
        vision,
        draft: None,
    };

    m.finish()?;
    definition
        .validate()
        .map_err(|error| FamilyError::invalid(error.to_string()))?;
    Ok(definition)
}

#[cfg(test)]
mod tests {
    use super::LoopLayout;

    #[test]
    fn loop_layout_rejects_mul_overflow() {
        let err = LoopLayout::new(
            6,
            12,
            u64::MAX,
            64,
            false,
            6,
            48,
            "inkling2.loop.core_end",
            "inkling2.num_loops",
        )
        .expect_err("multiplication overflow in extra_visits must return an error");
        assert!(err.to_string().contains("overflows"), "{err}");
    }

    #[test]
    fn loop_layout_rejects_add_overflow() {
        let err = LoopLayout::new(
            u64::MAX - 6,
            u64::MAX,
            2,
            64,
            false,
            1,
            u64::MAX,
            "inkling2.loop.core_end",
            "inkling2.num_loops",
        )
        .expect_err("addition overflow in core_end + extra_visits must return an error");
        assert!(err.to_string().contains("overflows"), "{err}");
    }

    #[test]
    fn loop_layout_rejects_exceeding_block_count_and_accepts_valid() {
        let err = LoopLayout::new(
            6,
            18,
            3,
            64,
            false,
            6,
            41,
            "inkling2.loop.core_end",
            "inkling2.num_loops",
        )
        .expect_err("core_end + extra_visits > block_count must fail");
        assert!(err.to_string().contains("exceeds block_count"), "{err}");

        let ok = LoopLayout::new(
            6,
            18,
            3,
            64,
            false,
            6,
            48,
            "inkling2.loop.core_end",
            "inkling2.num_loops",
        )
        .expect("valid looped layout");
        assert_eq!(ok.extra_visits(), 24);
        assert_eq!(ok.visit(6), (6, 0));
        assert_eq!(ok.visit(18), (6, 1));
        assert_eq!(ok.visit(42), (18, 0));
        assert_eq!(ok.first_visit(18), 42);
    }
}
