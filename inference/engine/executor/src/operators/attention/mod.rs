//! Softmax attention over history spans (`Operator::Attention`): the forms
//! the attention entries implement, their static axes, weight roles and
//! program binding.

pub(crate) mod graph;

use super::{PlanError, WeightPush, tail};
use crate::{AttentionBinding, AttentionShape, SelectionBinding, SelectionShape};
use magnitude_family_contracts::{
    Attention, AttentionGate, Decoder, GateFunction, GateGranularity, HeadNorm, HistoryDomain,
    HistoryReads, KeyValue, MediaRowAttention, Operator, OutputForm, Rotary, RotaryPair, ValueNorm,
    ValueSource, WeightKind,
};
use magnitude_state::KvCodec;
use seismic::Element;

/// The attention forms the attention entries implement: any output gate,
/// query and key norms both present or both absent, projected values or
/// values from the raw key, an optional unweighted value norm, one norm
/// epsilon, owned histories in a token or window domain or a Shared layer's
/// source history, and any rotary table whose stored divisors (checked at
/// load, `check_rotary_divisors`) cover its pairs. Media rows attend
/// causally or bidirectionally (`operators::media_rows_bidirectional`). A
/// history is read whole or selected by an indexer (`HistoryReads`).
pub(crate) fn admit(attention: &Attention, input_epsilon: f64) -> Result<(), PlanError> {
    // A Shared layer reads its source's history (`StateStore::history_read`)
    // and projects no keys or values; its source's domain is admitted with
    // the source.
    if let KeyValue::Owned {
        key_norm, domain, ..
    } = &attention.key_value
    {
        if matches!(attention.query_norm, HeadNorm::None) != matches!(key_norm, HeadNorm::None) {
            return Err(PlanError::Unsupported(
                "attention with only one of query and key norms",
            ));
        }
        if !matches!(domain, HistoryDomain::Token | HistoryDomain::Window { .. }) {
            return Err(PlanError::Unsupported("attention history domain"));
        }
    }
    head_norm_epsilon(attention, input_epsilon)?;
    if let AttentionGate::Interleaved {
        function: GateFunction::Silu,
    }
    | AttentionGate::Separate {
        function: GateFunction::Silu,
        ..
    } = &attention.gate
    {
        return Err(PlanError::Unsupported("SiLU attention gate"));
    }
    if let Rotary::Table {
        pairs,
        divisors: Some(divisors),
    } = &attention.rotary
    {
        if divisors.bases.len() != pairs.len() {
            return Err(PlanError::Topology(
                "rotary divisor bases disagree with the table's pairs",
            ));
        }
    }
    // A selected read's indexer orders keys by position (`indexer_select`):
    // its rows attend causally, and its norms share one epsilon.
    if let HistoryReads::Selected { indexer, .. } = &attention.reads {
        if attention.media_rows != MediaRowAttention::Causal {
            return Err(PlanError::Unsupported(
                "selected attention attending media rows bidirectionally",
            ));
        }
        if indexer.query_norm.epsilon != indexer.key_norm.epsilon {
            return Err(PlanError::Unsupported(
                "indexer norms with differing epsilons",
            ));
        }
    }
    let shape = shape(0, attention)?;
    if shape.width < 2 * shape.rotary_pairs {
        return Err(PlanError::Topology("rotary pairs exceed the head width"));
    }
    Ok(())
}

/// How the decoder's attention layers attend media rows, per history
/// domain: a launch row reads one fresh span per history read, so the owned
/// layers of one domain attend media rows alike, and a Shared layer like its
/// source (Gemma: bidirectional on its window layers, causal on its full
/// ones). Without a vision component there are no media rows to attend.
pub(crate) fn admit_media(decoder: &Decoder, vision: bool) -> Result<(), PlanError> {
    if !vision {
        return Ok(());
    }
    let mut domains: Vec<(&HistoryDomain, MediaRowAttention)> = Vec::new();
    for (index, sublayer) in decoder.sublayers() {
        let Operator::Attention(attention) = &sublayer.op else {
            continue;
        };
        let differ = || {
            PlanError::Unsupported(
                "attention layers of one history attending media rows differently",
            )
        };
        match &attention.key_value {
            KeyValue::Owned { domain, .. } => {
                match domains.iter().find(|(other, _)| *other == domain) {
                    Some((_, form)) if *form != attention.media_rows => return Err(differ()),
                    Some(_) => {}
                    None => domains.push((domain, attention.media_rows)),
                }
            }
            KeyValue::Shared { source } => {
                let Some(Operator::Attention(owner)) = decoder
                    .sublayers()
                    .find(|(candidate, _)| candidate == source)
                    .map(|(_, sublayer)| &sublayer.op)
                else {
                    return Err(PlanError::Topology(
                        "a Shared layer's source is not attention",
                    ));
                };
                if owner.media_rows != attention.media_rows || *source >= index {
                    return Err(differ());
                }
            }
        }
    }
    Ok(())
}

/// Check stored per-pair rotary divisors (`rope_freqs`) against the table the
/// family derived from headers: pair `p`'s frequency is `bases[p] / d_p`,
/// rounded to f32 where nonzero, and at most 1e-20 where the table leaves the
/// pair unrotated. The artifact is rejected otherwise.
pub(crate) fn check_rotary_divisors(
    pairs: &[RotaryPair],
    bases: &[f64],
    stored: &[f32],
) -> Result<(), String> {
    if stored.len() != pairs.len() || bases.len() != pairs.len() {
        return Err(format!(
            "{} stored rotary divisors for {} pairs",
            stored.len(),
            pairs.len()
        ));
    }
    for (index, ((pair, base), divisor)) in pairs.iter().zip(bases).zip(stored).enumerate() {
        let frequency = base / f64::from(*divisor);
        let matches = if pair.frequency == 0.0 {
            frequency <= 1e-20
        } else {
            frequency as f32 == pair.frequency as f32
        };
        if !matches {
            return Err(format!(
                "rotary pair {index}: the stored divisor {divisor} gives frequency {frequency}, \
                 the table {}",
                pair.frequency
            ));
        }
    }
    Ok(())
}

/// The attention entries' static axes of an attention operator over a
/// `hidden`-wide residual.
pub(crate) fn shape(hidden: u64, attention: &Attention) -> Result<AttentionShape, PlanError> {
    if attention.kv_heads == 0 || attention.heads % attention.kv_heads != 0 {
        return Err(PlanError::Topology(
            "attention heads are not a multiple of its key/value heads",
        ));
    }
    let width = attention.width;
    let (interleaved_gate, separate_gate) = match &attention.gate {
        AttentionGate::None => (0, 0),
        AttentionGate::Interleaved { .. } => (width, 0),
        AttentionGate::Separate { granularity, .. } => match granularity {
            GateGranularity::Element => (0, width),
            GateGranularity::Head => (0, 1),
        },
    };
    let (fresh, value_norm, projected_value) = match &attention.key_value {
        KeyValue::Owned {
            value, value_norm, ..
        } => (
            1,
            u64::from(matches!(value_norm, ValueNorm::RmsUnweighted(_))),
            matches!(value, ValueSource::Projected(_)),
        ),
        KeyValue::Shared { .. } => (0, 0, false),
    };
    Ok(AttentionShape {
        hidden,
        kv_heads: attention.kv_heads,
        group: attention.heads / attention.kv_heads,
        rotary_pairs: match &attention.rotary {
            Rotary::None => 0,
            Rotary::Interleaved { width, .. } => width / 2,
            Rotary::Table { pairs, .. } => pairs.len() as u64,
        },
        width,
        interleaved_gate,
        separate_gate,
        fresh,
        head_norm: u64::from(matches!(attention.query_norm, HeadNorm::Rms(_))),
        value_norm,
        projected_value,
    })
}

/// The attention entries' `gate_function` code: 0 sigmoid, 1 softplus (and
/// 0 for an ungated operator, whose gate is never read). [`admit`] rejects
/// every other gate.
pub(crate) fn gate_function(attention: &Attention) -> i32 {
    let function = match &attention.gate {
        AttentionGate::None => return 0,
        AttentionGate::Interleaved { function } | AttentionGate::Separate { function, .. } => {
            function
        }
    };
    match function {
        GateFunction::Sigmoid => 0,
        GateFunction::Softplus => 1,
        GateFunction::Silu => unreachable!("admission rejects SiLU attention gates"),
    }
}

/// The one epsilon of an attention operator's query, key and value norms,
/// which the attention entries take as one argument. An operator without
/// head norms uses its input norm's (never read).
pub(crate) fn head_norm_epsilon(
    attention: &Attention,
    input_epsilon: f64,
) -> Result<f64, PlanError> {
    let mut epsilons = Vec::new();
    if let HeadNorm::Rms(norm) = &attention.query_norm {
        epsilons.push(norm.epsilon);
    }
    if let KeyValue::Owned {
        key_norm,
        value_norm,
        ..
    } = &attention.key_value
    {
        if let HeadNorm::Rms(norm) = key_norm {
            epsilons.push(norm.epsilon);
        }
        if let ValueNorm::RmsUnweighted(norm) = value_norm {
            epsilons.push(norm.epsilon);
        }
    }
    match epsilons.split_first() {
        None => Ok(input_epsilon),
        Some((first, rest)) if rest.iter().all(|epsilon| epsilon == first) => Ok(*first),
        Some(_) => Err(PlanError::Unsupported(
            "attention head norms with differing epsilons",
        )),
    }
}

/// The weight kind of an attention operator's query projection: fused with
/// the gate rows when the gate is interleaved.
pub(crate) fn query_kind(attention: &Attention) -> WeightKind {
    match attention.gate {
        AttentionGate::Interleaved { .. } => WeightKind::QueryGate,
        AttentionGate::None | AttentionGate::Separate { .. } => WeightKind::Query,
    }
}

/// The weights the operator binds, in the order its entries consume them.
pub(super) fn weights<'a>(attention: &'a Attention, push: &mut WeightPush<'_, 'a>) {
    push(query_kind(attention), &attention.query);
    if let AttentionGate::Separate { weight, .. } = &attention.gate {
        push(WeightKind::AttentionGate, weight);
    }
    if let KeyValue::Owned { key, value, .. } = &attention.key_value {
        push(WeightKind::Key, key);
        if let ValueSource::Projected(value) = value {
            push(WeightKind::Value, value);
        }
    }
    if let HeadNorm::Rms(norm) = &attention.query_norm {
        push(WeightKind::QueryNorm, &norm.weight);
    }
    if let Rotary::Table {
        divisors: Some(divisors),
        ..
    } = &attention.rotary
    {
        push(WeightKind::RotaryDivisors, &divisors.weight);
    }
    if let KeyValue::Owned {
        key_norm: HeadNorm::Rms(norm),
        ..
    } = &attention.key_value
    {
        push(WeightKind::KeyNorm, &norm.weight);
    }
    if let HistoryReads::Selected { indexer, .. } = &attention.reads {
        push(WeightKind::IndexerQuery, &indexer.query);
        push(WeightKind::IndexerKey, &indexer.key);
        push(WeightKind::IndexerQueryNorm, &indexer.query_norm.weight);
        push(WeightKind::IndexerKeyNorm, &indexer.key_norm.weight);
    }
    push(WeightKind::AttentionOutput, &attention.output);
}

/// The static axes of a selected read in a decoder serving `context`
/// tokens; none for a read of every visible row.
pub(crate) fn selection_shape(reads: &HistoryReads, context: u64) -> Option<SelectionShape> {
    match reads {
        HistoryReads::Visible => None,
        HistoryReads::Selected {
            block,
            blocks,
            indexer,
        } => Some(SelectionShape {
            heads: indexer.heads,
            width: indexer.width,
            block: *block,
            blocks: *blocks,
            // A row at position t < context reads blocks up to its tail's,
            // t / block: the bitmap covers them all.
            words: context / block / 32 + 1,
        }),
    }
}

/// A selected read's numerical parameters apart from its weights: block,
/// blocks, indexer heads, width and norm epsilon.
fn selection_key(reads: &HistoryReads) -> Option<(u64, u64, u64, u64, f64)> {
    match reads {
        HistoryReads::Visible => None,
        HistoryReads::Selected {
            block,
            blocks,
            indexer,
        } => Some((
            *block,
            *blocks,
            indexer.heads,
            indexer.width,
            indexer.query_norm.epsilon,
        )),
    }
}

/// Every numerical parameter of the operator a sealed graph holds apart from
/// its weights.
pub(super) fn shape_key(attention: &Attention, epsilon: f64) -> Result<String, PlanError> {
    Ok(format!(
        "attention {:?} {:?} {:?} {:?} {} {:?}",
        shape(0, attention)?,
        gate_function(attention),
        head_norm_epsilon(attention, epsilon)?,
        attention.rotary,
        attention.scale,
        selection_key(&attention.reads)
    ))
}

/// The program binding of an attention sublayer with `output` in a decoder
/// serving `context` tokens; `lookup` resolves the planned element of a role
/// in its scope. An absent projection segment binds a zero-row view of the
/// query weight.
pub(super) fn binding(
    attention: &Attention,
    output: &OutputForm,
    hidden: u64,
    context: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    activation: Element,
    history: KvCodec,
) -> Result<AttentionBinding, PlanError> {
    let shape = shape(hidden, attention)?;
    let query = lookup(query_kind(attention))?;
    let segment = |present: bool, kind| if present { lookup(kind) } else { Ok(query) };
    let selection = selection_shape(&attention.reads, context)
        .map(|shape| {
            Ok::<_, PlanError>(SelectionBinding {
                shape,
                query: lookup(WeightKind::IndexerQuery)?,
                key: lookup(WeightKind::IndexerKey)?,
            })
        })
        .transpose()?;
    Ok(AttentionBinding {
        selection,
        key_value_only: false,
        shape,
        norm: lookup(WeightKind::InputNorm)?,
        query,
        gate: segment(shape.gate_rows() > 0, WeightKind::AttentionGate)?,
        key: segment(shape.key_rows() > 0, WeightKind::Key)?,
        value: segment(shape.value_rows() > 0, WeightKind::Value)?,
        output: lookup(WeightKind::AttentionOutput)?,
        activation,
        history: history
            .spec(
                activation
                    .dtype()
                    .ok_or(PlanError::Unsupported("attention activation dtype"))?,
                shape.width as usize,
                shape.width as usize,
            )
            .validate()
            .map_err(|error| PlanError::InvalidDefinition(error.to_string()))?,
        tail: tail(output, &lookup)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(frequency: f64) -> RotaryPair {
        RotaryPair {
            frequency,
            amplitude: 1.0,
        }
    }

    mod selected {
        use crate::programs::native_target_graph::checked_target_family_storage;
        use crate::{
            ComponentSelection, ExecutionPath, MixerProgramSlot, ModelLoadPlan, PlannedMethod,
            ResourceCapacity, ResourceLimits, ResourcePlanner, SelectionShape, resident_layout,
            source_element,
        };
        use magnitude_artifacts::{
            ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest,
            gguf::{Encoding, TensorDescriptor},
        };
        use magnitude_family_contracts::{
            ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder,
            DenseFfn, EmbeddingScale, EntryForm, ExitForm, ExitNorm, FamilyId, FeedForwardUp,
            HeadNorm, HistoryDomain, HistoryReads, Indexer, InputNorm, InputSemantics, KeyValue,
            MediaRowAttention, ModelDefinition, Operator, OutputForm, ResidualForm, RmsNorm,
            Rotary, RotaryPair, Sublayer, ValueNorm, ValueSource, WeightDescriptor,
        };
        use magnitude_state::{HistoryDomainKind, KvCodec, LayerRef};
        use seismic::BackendName;

        const HIDDEN: u64 = 256;
        const CONTEXT: u64 = 4096;

        /// A decoder of two blocks, each a selected attention (64-wide heads,
        /// fully rotated; an indexer of two 64-wide heads over blocks of four
        /// rows, two selected) and a dense feed-forward.
        fn selected() -> (ModelDefinition, PackageManifest) {
            let mut tensors = Vec::new();
            let mut offset = 0;
            let mut tensor = |name: String, shape: Vec<u64>| {
                let encoding = if shape.len() > 1 {
                    Encoding::BF16
                } else {
                    Encoding::F32
                };
                let nbytes = source_element(encoding, *shape.last().unwrap())
                    .unwrap()
                    .canonical_byte_len(&shape)
                    .unwrap();
                tensors.push(TensorDescriptor {
                    name: name.clone(),
                    shape: shape.clone(),
                    encoding,
                    offset,
                    nbytes,
                });
                offset += nbytes;
                WeightDescriptor::stored(name, shape)
            };
            let rms = |weight| RmsNorm {
                weight,
                epsilon: 1e-6,
            };
            let mut block = |layer: usize| {
                let name = |suffix: &str| format!("blk.{layer}.{suffix}");
                let attention = Sublayer {
                    input: InputNorm::Rms(rms(tensor(name("attn_norm"), vec![HIDDEN]))),
                    op: Operator::Attention(Box::new(Attention {
                        heads: 4,
                        kv_heads: 2,
                        width: 64,
                        query: tensor(name("attn_q"), vec![HIDDEN, HIDDEN]),
                        gate: AttentionGate::None,
                        query_norm: HeadNorm::Rms(rms(tensor(name("attn_q_norm"), vec![64]))),
                        key_value: KeyValue::Owned {
                            key: tensor(name("attn_k"), vec![128, HIDDEN]),
                            value: ValueSource::Projected(tensor(
                                name("attn_v"),
                                vec![128, HIDDEN],
                            )),
                            key_norm: HeadNorm::Rms(rms(tensor(name("attn_k_norm"), vec![64]))),
                            value_norm: ValueNorm::None,
                            domain: HistoryDomain::Token,
                        },
                        rotary: Rotary::Table {
                            pairs: (0..32)
                                .map(|pair| RotaryPair {
                                    frequency: 1.0e4f64.powf(-(2 * pair) as f64 / 64.0),
                                    amplitude: 1.0,
                                })
                                .collect(),
                            divisors: None,
                        },
                        scale: 1.0 / 8.0,
                        reads: HistoryReads::Selected {
                            block: 4,
                            blocks: 2,
                            indexer: Indexer {
                                heads: 2,
                                width: 64,
                                query: tensor(name("indexer.q_proj"), vec![128, HIDDEN]),
                                key: tensor(name("indexer.k_proj"), vec![64, HIDDEN]),
                                query_norm: rms(tensor(name("indexer.q_norm"), vec![64])),
                                key_norm: rms(tensor(name("indexer.k_norm"), vec![64])),
                            },
                        },
                        media_rows: MediaRowAttention::Causal,
                        output: tensor(name("attn_output"), vec![HIDDEN, HIDDEN]),
                    })),
                    output: OutputForm::Residual,
                };
                let dense = Sublayer {
                    input: InputNorm::Rms(rms(tensor(name("ffn_norm"), vec![HIDDEN]))),
                    op: Operator::DenseFfn(Box::new(DenseFfn {
                        intermediate: 512,
                        up: FeedForwardUp::Gated {
                            activation: ActivationFunction::Silu,
                            gate: tensor(name("ffn_gate"), vec![512, HIDDEN]),
                            up: tensor(name("ffn_up"), vec![512, HIDDEN]),
                        },
                        down: tensor(name("ffn_down"), vec![HIDDEN, 512]),
                    })),
                    output: OutputForm::Residual,
                };
                Block {
                    sublayers: vec![attention, dense],
                }
            };
            let blocks = vec![block(0), block(1)];
            let embedding = tensor("token_embd".into(), vec![512, HIDDEN]);
            let output_norm = tensor("output_norm".into(), vec![HIDDEN]);
            let identity = PackageIdentity {
                target: ArtifactIdentity([7; 32]),
                projector: None,
            };
            let definition = ModelDefinition {
                family: FamilyId("selected-attention".into()),
                artifact_identity: identity,
                inputs: InputSemantics { coordinate_axes: 1 },
                decoder: Decoder {
                    activation_dtype: ActivationDType::BF16,
                    hidden: HIDDEN,
                    vocabulary: 512,
                    context_limit: CONTEXT,
                    residual: ResidualForm::Single,
                    entry: EntryForm {
                        embedding: embedding.clone(),
                        scale: EmbeddingScale::Unit,
                        norm: None,
                        per_layer: None,
                        hash_routing: None,
                    },
                    blocks,
                    exit: ExitForm {
                        norm: ExitNorm::Rms(rms(output_norm)),
                        output: embedding,
                        softcap: None,
                    },
                },
                head: None,
                vision: None,
                draft: None,
            };
            let manifest = PackageManifest {
                identity,
                target: ComponentManifest {
                    files: vec![ComponentFile {
                        path: "selected.gguf".into(),
                        size: offset,
                    }],
                    identity: identity.target,
                    tensors,
                },
                projector: None,
                draft: None,
            };
            (definition, manifest)
        }

        /// A selected read binds its indexer at the served context, keeps
        /// its raw keys and block keys in their own domains, and its graphs
        /// (the checked metadata route of the production fragment: the
        /// indexer, the Segments decode and the WorkList prefill) certify on
        /// every backend at every row class.
        #[test]
        fn selected_attention_plans_its_indexer_and_certifies_on_every_backend() {
            let (definition, manifest) = selected();
            definition.validate().unwrap();
            assert_eq!(definition.deferred_forms(), []);
            let limits = ResourceLimits {
                max_launch_rows: 64,
                max_launch_slots: 8,
                max_selected_rows: 64,
                max_drafting_slots: 8,
                exported_logits_rows: 0,
                max_images_per_request: 1,
                max_image_cells: 0,
                lookahead: false,
                group_capacity: std::num::NonZeroUsize::MIN,
            };
            for backend in [
                BackendName::Cpu,
                BackendName::Metal,
                BackendName::Cuda,
                BackendName::Vulkan,
            ] {
                for codec in [KvCodec::Dense, KvCodec::AffineK8V4] {
                    let load = ModelLoadPlan::derive(
                        &manifest,
                        &definition,
                        ComponentSelection {
                            head: false,
                            vision: false,
                        },
                        resident_layout(ExecutionPath::Native, backend),
                    )
                    .unwrap();
                    let plan = load.program_plan(&definition, codec).unwrap();
                    let MixerProgramSlot::Attention(binding) = plan.target().blocks()[0].mixer()
                    else {
                        panic!("an attention block");
                    };
                    assert_eq!(
                        binding.selection.map(|selection| selection.shape),
                        Some(SelectionShape {
                            heads: 2,
                            width: 64,
                            block: 4,
                            blocks: 2,
                            // Blocks [0, 1024] of a 4096-token context.
                            words: 33,
                        })
                    );
                    let state = ResourcePlanner::state_plan(
                        &definition,
                        &load,
                        PlannedMethod::Plain,
                        codec,
                        limits,
                        ResourceCapacity {
                            domain_bytes: 16 * 1024 * 1024 * 1024,
                            tensor_operations: crate::TensorOperations::Absent,
                        },
                    )
                    .unwrap();
                    // The Token domain, then each indexer's raw keys and
                    // block keys.
                    let kinds = state
                        .target_state()
                        .domains
                        .iter()
                        .map(|domain| domain.kind)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        kinds,
                        [
                            HistoryDomainKind::Token,
                            HistoryDomainKind::Window { rows: 3 },
                            HistoryDomainKind::Block { rate: 4 },
                        ]
                    );
                    for block in 0..2 {
                        let (window, blocks) =
                            state.target_state().indexer_history(block).unwrap();
                        assert!(
                            window
                                .components
                                .iter()
                                .any(|component| component.layer == LayerRef::Indexer(block))
                        );
                        assert!(blocks.rows as u64 >= CONTEXT / 4);
                    }
                    checked_target_family_storage(
                        backend,
                        &load,
                        &definition.decoder,
                        &state,
                        plan.target(),
                        limits,
                    )
                    .unwrap_or_else(|error| panic!("{backend:?} {codec:?}: {error}"));
                }
            }
        }
    }

    #[test]
    fn stored_rotary_divisors_must_reproduce_the_table() {
        let bases = [1.0, 0.5, 0.25];
        let pairs = [pair(1.0), pair(0.25), pair(0.0)];
        // 1/1, 0.5/2, 0.25/1e30 (unrotated).
        check_rotary_divisors(&pairs, &bases, &[1.0, 2.0, 1e30]).unwrap();
        assert!(check_rotary_divisors(&pairs, &bases, &[1.0, 4.0, 1e30]).is_err());
        assert!(check_rotary_divisors(&pairs, &bases, &[1.0, 2.0, 1.0]).is_err());
        assert!(check_rotary_divisors(&pairs, &bases, &[1.0, 2.0]).is_err());
    }
}
