use crate::{
    BankComponent, ComponentDescriptor, ComponentSpec, HistoryDomainLayout, KvCodec, LayerRef,
};
use magnitude_family_contracts::{
    ActivationDType, Attention, Block, Decoder, GatedDelta, HistoryDomain, HistoryReads, Indexer,
    KeyValue, NGramInput, Operator, ResidualForm, ShortConv, StateSpace, SublayerIndex,
};
use seismic::DType;

pub const SLAB_BYTE_TARGET: u64 = 64 * 1024 * 1024;
pub const SLAB_ROW_TILE: usize = 256;
/// The most spans any history of a domain can present to one launch, so the
/// largest span class a store seals is 64.
pub const MAX_HISTORY_SPANS: usize = 63;

/// How one stored history domain places rows (see [`history_geometry`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryGeometry {
    /// Rows of one page: the unit a history takes beyond its own tail page.
    pub page_rows: usize,
    /// Rows of one slab: a whole number of pages.
    pub slab_rows: usize,
    /// The most spans a history of the domain presents to a launch.
    pub span_limit: usize,
}

impl HistoryGeometry {
    /// The rows a domain reserves for `rows` requested: whole pages, so every
    /// page is complete and a history's page demand is exactly what its
    /// claims take.
    pub fn reserved_rows(&self, rows: usize) -> Option<usize> {
        rows.checked_next_multiple_of(self.page_rows)
    }
}

/// The placement geometry of a stored history domain whose histories hold at
/// most `row_limit` rows. `appended_rows` is the most rows one advance
/// appends that Shared readers of the domain see as history (0 without
/// readers).
///
/// A history takes rows only in place after its end within its last page, or
/// as whole free pages, so every page it references is complete except its
/// first and last. It therefore presents at most `ceil(row_limit / page) + 1`
/// spans, plus `ceil(appended_rows / page) + 1` for rows a Shared or causal compressed target reader sees
/// appended within an advance. Pages are the smallest multiple of
/// [`SLAB_ROW_TILE`] keeping that at most [`MAX_HISTORY_SPANS`]; a slab holds
/// as many whole pages as fit [`SLAB_BYTE_TARGET`], and at least one.
pub fn history_geometry(
    row_bytes: u64,
    row_limit: usize,
    appended_rows: usize,
) -> Result<HistoryGeometry, String> {
    if row_bytes == 0 {
        return Err("history row must have storage".into());
    }
    let span_limit = |page_rows: usize| {
        let appended = if appended_rows == 0 {
            0
        } else {
            appended_rows.div_ceil(page_rows) + 1
        };
        row_limit.div_ceil(page_rows) + 1 + appended
    };
    // No page below this many tiles meets the limit (each span holds at most
    // a page), so the search starts there and settles within a few tiles.
    let mut tiles = ((row_limit + appended_rows) / (MAX_HISTORY_SPANS * SLAB_ROW_TILE)).max(1);
    while span_limit(tiles * SLAB_ROW_TILE) > MAX_HISTORY_SPANS {
        tiles += 1;
    }
    let page_rows = tiles * SLAB_ROW_TILE;
    let slab_pages = usize::try_from(SLAB_BYTE_TARGET / row_bytes / page_rows as u64)
        .map_err(|_| "history slab row count exceeds host domain")?
        .max(1);
    Ok(HistoryGeometry {
        page_rows,
        slab_rows: slab_pages
            .checked_mul(page_rows)
            .ok_or("history slab row count exceeds host domain")?,
        span_limit: span_limit(page_rows),
    })
}

/// Banks in one recurrent slab. A bank larger than the byte target still
/// occupies one slab by itself.
pub fn banks_per_slab(bank_bytes: u64) -> Result<usize, String> {
    if bank_bytes == 0 {
        return Err("recurrent bank must have storage".into());
    }
    usize::try_from((SLAB_BYTE_TARGET / bank_bytes).max(1))
        .map_err(|_| "recurrent slab bank count exceeds host domain".into())
}

/// Device-free state allocation plan derived from the family-neutral model
/// geometry. It is shared by every execution path; only the physical stage
/// that binds the resulting planes differs.
///
/// Target history is grouped into history domains: one Token domain, one
/// Window domain per distinct window, and one Shared domain per source layer,
/// in that order (each only when present), then, with selected attention,
/// its indexers' raw-key Window and block-key Block domains. Recurrent banks list each
/// recurrent block's components in block order: gated delta
/// `[window, state, tape]`, short convolution `[window]`, state space
/// `[window, state, tape]`, then an n-gram input's `[window]` after its
/// block's mixer's. `target_recurrent` is their physical form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelStateLayout {
    pub target_history: Vec<HistoryDomainLayout>,
    pub target_banks: Vec<BankComponent>,
    pub target_recurrent: Vec<ComponentSpec>,
    pub head_history: Vec<HistoryDomainLayout>,
}

impl ModelStateLayout {
    /// `drafter` is the blocks of the selected drafter, if any (an embedded
    /// head's blocks, or a separate draft's layers). `tape_rows` is the most
    /// speculative rows a recurrent bank records after its published state
    /// (the planned draft width; 0 without drafting).
    ///
    /// A target history component is named by its decoder block and a
    /// drafter component by its drafter block; a block holds at most one
    /// stateful mixer, and its n-gram input's bank follows the mixer's.
    /// Drafter history is grouped into domains as the
    /// target's is.
    pub fn derive(
        decoder: &Decoder,
        drafter: &[&Block],
        target_codec: KvCodec,
        tape_rows: usize,
    ) -> Result<Self, String> {
        let activation = activation_dtype(decoder.activation_dtype);
        let mut target_history = HistoryDomains::default();
        let mut target_banks = Vec::new();
        for (index, block) in decoder.blocks.iter().enumerate() {
            let layer = LayerRef::Target(layer(index)?);
            for stateful in block_state(block)? {
                match stateful {
                    Stateful::History(attention, domain) => {
                        target_history.own(
                            domain,
                            history_component(attention, layer, target_codec, activation)?,
                        )?;
                        if let HistoryReads::Selected { block, indexer, .. } = &attention.reads {
                            target_history.index(index, *block, indexer)?;
                        }
                    }
                    Stateful::Shared(source) => {
                        target_history.share(LayerRef::Target(source.block), layer)
                    }
                    Stateful::GatedDelta(delta) => {
                        target_banks.extend(delta_bank(delta, activation)?)
                    }
                    Stateful::ShortConv(convolution) => {
                        target_banks.push(short_convolution_bank(convolution)?)
                    }
                    Stateful::StateSpace(space) => {
                        target_banks.extend(state_space_bank(space, activation)?)
                    }
                    Stateful::NGram(input) => target_banks.extend(ngram_bank(decoder, input)?),
                }
            }
        }
        let target_recurrent = target_banks
            .iter()
            .map(|component| component.spec(tape_rows))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;

        let mut head_history = HistoryDomains::default();
        for (index, block) in drafter.iter().enumerate() {
            match block_state(block)?.as_slice() {
                // Drafter history stays dense whatever the target codec: the
                // drafters' attention reads dense planes only.
                [Stateful::History(attention, domain)] => head_history.own(
                    *domain,
                    history_component(
                        attention,
                        LayerRef::Head(layer(index)?),
                        KvCodec::Dense,
                        activation,
                    )?,
                )?,
                _ => {
                    return Err(
                        "a drafter block must hold exactly one owned attention history".into(),
                    );
                }
            }
        }

        Ok(Self {
            target_history: target_history.into_layouts(),
            target_banks,
            target_recurrent,
            head_history: head_history.into_layouts(),
        })
    }
}

/// Target history components grouped by domain as blocks are visited.
#[derive(Default)]
struct HistoryDomains {
    token: Vec<ComponentDescriptor>,
    windows: Vec<(usize, Vec<ComponentDescriptor>)>,
    shared: Vec<(LayerRef, Vec<LayerRef>)>,
    /// Selected attention's indexers: their block size, their raw keys
    /// (the incomplete block's, a Window of `block - 1` rows) and their
    /// finished block keys (a Block domain of that rate).
    indexers: Option<(usize, Vec<ComponentDescriptor>, Vec<ComponentDescriptor>)>,
}

impl HistoryDomains {
    fn own(&mut self, domain: HistoryDomain, component: ComponentDescriptor) -> Result<(), String> {
        match domain {
            HistoryDomain::Token => self.token.push(component),
            HistoryDomain::Window { tokens } => {
                let rows = host(tokens, "attention window")?;
                match self.windows.iter_mut().find(|(window, _)| *window == rows) {
                    Some((_, components)) => components.push(component),
                    None => self.windows.push((rows, vec![component])),
                }
            }
            HistoryDomain::Block { .. } => {
                return Err("block history domains are not supported".into());
            }
        }
        Ok(())
    }

    /// Block `index`'s indexer: one key row of `indexer.width` F32 values
    /// (the indexer scores in F32, as the reference does) per token, with
    /// its coordinates, in the raw-key domain and per block in the
    /// block-key domain. Every indexer of a decoder shares one block size.
    fn index(&mut self, index: usize, block: u64, indexer: &Indexer) -> Result<(), String> {
        let rate = host(block, "selection block")?;
        if rate < 2 {
            return Err("a selection block holds at least two rows".into());
        }
        let component = |coordinates| {
            ComponentDescriptor::indexer(
                LayerRef::Indexer(layer(index)?),
                DType::F32,
                host(indexer.width, "indexer width")?,
                coordinates,
            )
            .map_err(|error| error.to_string())
        };
        let (block, raw, finished) = self
            .indexers
            .get_or_insert_with(|| (rate, Vec::new(), Vec::new()));
        if *block != rate {
            return Err("selected attention layers with differing blocks".into());
        }
        raw.push(component(true)?);
        finished.push(component(false)?);
        Ok(())
    }

    fn share(&mut self, source: LayerRef, layer: LayerRef) {
        match self.shared.iter_mut().find(|(owner, _)| *owner == source) {
            Some((_, layers)) => layers.push(layer),
            None => self.shared.push((source, vec![layer])),
        }
    }

    fn into_layouts(self) -> Vec<HistoryDomainLayout> {
        let token = (!self.token.is_empty()).then_some(HistoryDomainLayout::Token {
            components: self.token,
        });
        token
            .into_iter()
            .chain(
                self.windows
                    .into_iter()
                    .map(|(rows, components)| HistoryDomainLayout::Window { rows, components }),
            )
            .chain(
                self.shared
                    .into_iter()
                    .map(|(source, layers)| HistoryDomainLayout::Shared { source, layers }),
            )
            .chain(self.indexers.into_iter().flat_map(|(rate, raw, finished)| {
                [
                    HistoryDomainLayout::Window {
                        rows: rate - 1,
                        components: raw,
                    },
                    HistoryDomainLayout::Block {
                        rate,
                        components: finished,
                    },
                ]
            }))
            .collect()
    }
}

/// The state one stateful sublayer keeps.
enum Stateful<'a> {
    History(&'a Attention, HistoryDomain),
    Shared(SublayerIndex),
    GatedDelta(&'a GatedDelta),
    ShortConv(&'a ShortConv),
    StateSpace(&'a StateSpace),
    NGram(&'a NGramInput),
}

/// The state a block keeps, in sublayer order: at most one mixer's (an
/// attention history or a recurrent bank), and the n-gram input's bank.
fn block_state(block: &Block) -> Result<Vec<Stateful<'_>>, String> {
    let mut state = Vec::new();
    let mut mixers = 0;
    for sublayer in &block.sublayers {
        let stateful = match &sublayer.op {
            // Stored rows hold the key and the value whatever the value's
            // source: a raw-key value is taken before key norm and rotary.
            Operator::Attention(attention) => match &attention.key_value {
                KeyValue::Owned { domain, .. } => Stateful::History(attention, *domain),
                KeyValue::Shared { source } => Stateful::Shared(*source),
            },
            Operator::GatedDelta(delta) => Stateful::GatedDelta(delta),
            Operator::ShortConv(convolution) => Stateful::ShortConv(convolution),
            Operator::StateSpace(space) => Stateful::StateSpace(space),
            Operator::NGramInput(input) => {
                state.push(Stateful::NGram(input));
                continue;
            }
            Operator::DenseFfn(_) | Operator::RoutedFfn(_) | Operator::PerLayerInput(_) => continue,
            Operator::Parallel(branches)
                if branches.iter().all(|branch| {
                    matches!(
                        branch.op,
                        Operator::DenseFfn(_) | Operator::RoutedFfn(_) | Operator::PerLayerInput(_)
                    )
                }) =>
            {
                continue;
            }
            op @ (Operator::LatentAttention(_) | Operator::Parallel(_)) => {
                return Err(format!("{} state has no layout yet", op.name()));
            }
        };
        mixers += 1;
        if mixers > 1 {
            return Err("a block holds more than one stateful mixer".into());
        }
        state.push(stateful);
    }
    Ok(state)
}

fn history_component(
    attention: &Attention,
    layer: LayerRef,
    codec: KvCodec,
    activation: DType,
) -> Result<ComponentDescriptor, String> {
    let kv_heads = host(attention.kv_heads, "attention KV heads")?;
    let width = host(attention.width, "attention head width")?;
    ComponentDescriptor::new(layer, codec.spec(activation, width, width), kv_heads)
        .map_err(|error| error.to_string())
}

fn activation_dtype(dtype: ActivationDType) -> DType {
    match dtype {
        ActivationDType::F16 => DType::F16,
        ActivationDType::BF16 => DType::BF16,
    }
}

/// A gated delta layer's bank components, in the order the state entries
/// take them (`recurrent.seismic`): the window (C - 1 raw rows before the
/// state, then the raw rows of the tape), the delta state, and the tape of
/// the rows after the state (innovations, keys, decays).
fn delta_bank(recurrent: &GatedDelta, activation: DType) -> Result<[BankComponent; 3], String> {
    let width = host(recurrent.width, "recurrent head width")?;
    let value_heads = host(recurrent.value_heads, "recurrent value heads")?;
    // The window stores raw projection rows, already rounded, in activation
    // precision.
    Ok([
        BankComponent::ConvWindow {
            width: host(recurrent.convolution_width, "recurrent convolution width")?,
            channels: host(
                recurrent.channels().map_err(|error| error.to_string())?,
                "recurrent channels",
            )?,
            dtype: activation,
        },
        BankComponent::DeltaState {
            heads: value_heads,
            width,
        },
        BankComponent::DeltaTape {
            value_heads,
            key_heads: host(recurrent.key_heads, "recurrent key heads")?,
            width,
        },
    ])
}

/// The n-gram input's bank components, in the order its entries take them:
/// its window of u rows (`ngram_inject`), F32, over the decoder's S * H
/// stream channels, `dilation (kernel - 1)` rows before the state (a dense
/// window of that reach plus one); then its token window (`ngram_rows`),
/// the sequence's `context` tokens before the state, each stored as
/// token + 1 in one U32 channel so the zero seed holds no token.
fn ngram_bank(decoder: &Decoder, input: &NGramInput) -> Result<[BankComponent; 2], String> {
    let ResidualForm::HyperConnections(streams) = &decoder.residual else {
        return Err("an n-gram input without residual streams".into());
    };
    let reach = input
        .dilation
        .checked_mul(
            input
                .kernel
                .checked_sub(1)
                .ok_or("n-gram kernel must be positive")?,
        )
        .ok_or("n-gram reach overflows")?;
    Ok([
        BankComponent::ConvWindow {
            width: host(reach + 1, "n-gram convolution reach")?,
            channels: host(
                streams
                    .streams
                    .checked_mul(decoder.hidden)
                    .ok_or("n-gram channels overflow")?,
                "n-gram channels",
            )?,
            dtype: DType::F32,
        },
        BankComponent::ConvWindow {
            width: host(
                input
                    .hash
                    .context
                    .checked_add(1)
                    .ok_or("n-gram context overflows")?,
                "n-gram context",
            )?,
            channels: 1,
            dtype: DType::U32,
        },
    ])
}

/// A short convolution's bank: its window of `u = B⊙X` rows, stored F32 so
/// the product is not rounded before the taps' sum.
fn short_convolution_bank(convolution: &ShortConv) -> Result<BankComponent, String> {
    Ok(BankComponent::ConvWindow {
        width: host(convolution.width, "short convolution width")?,
        channels: host(convolution.channels, "short convolution channels")?,
        dtype: DType::F32,
    })
}

/// A Mamba-2 layer's bank: the window of raw `xBC` projection rows in
/// activation precision, the F32 state, and the tape of the additive rule.
fn state_space_bank(space: &StateSpace, activation: DType) -> Result<[BankComponent; 3], String> {
    let heads = host(space.heads, "state space heads")?;
    let head_width = host(space.head_width, "state space head width")?;
    let state_width = host(space.state, "state space state width")?;
    Ok([
        BankComponent::ConvWindow {
            width: host(space.convolution_width, "state space convolution width")?,
            channels: host(
                space.channels().map_err(|error| error.to_string())?,
                "state space channels",
            )?,
            dtype: activation,
        },
        BankComponent::SsmState {
            heads,
            head_width,
            state_width,
        },
        BankComponent::SsmTape {
            heads,
            head_width,
            groups: host(space.groups, "state space groups")?,
            state_width,
        },
    ])
}

fn host(value: u64, field: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{field} exceeds the host domain"))
}

fn layer(index: usize) -> Result<u32, String> {
    u32::try_from(index).map_err(|_| "model layer index exceeds u32".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_family_contracts::{
        AttentionGate, EmbeddingScale, EntryForm, ExitForm, ExitNorm, GateFunction, HeadNorm,
        HistoryReads, InputNorm, MediaRowAttention, OutputForm, RecurrentHeadMapping, ResidualForm,
        RmsNorm, Rotary, Sublayer, ValueNorm, ValueSource, WeightDescriptor,
    };

    #[test]
    fn history_geometry_is_the_smallest_page_within_the_span_limit() {
        // Qwen3.8-27B (16 K8/V4 layers x 4 heads) at 65,792 rows: the
        // incident geometry.
        let qwen38 = history_geometry(28_672, 65_792, 0).unwrap();
        assert_eq!(qwen38.page_rows, 1_280);
        assert_eq!(qwen38.span_limit, 53);
        assert_eq!(qwen38.slab_rows % qwen38.page_rows, 0);
        assert!(qwen38.slab_rows as u64 * 28_672 <= SLAB_BYTE_TARGET);
        // Qwen3.6-35B (10 layers x 2 heads).
        let qwen36 = history_geometry(8_960, 65_792, 0).unwrap();
        assert_eq!((qwen36.page_rows, qwen36.slab_rows), (1_280, 6_400));
        // A short window keeps one-tile pages; a row above the byte target
        // still gets a slab of one page.
        let window = history_geometry(SLAB_BYTE_TARGET, 1_024 + 512, 0).unwrap();
        assert_eq!((window.page_rows, window.slab_rows), (256, 256));
        // Shared readers see an advance's appended rows too.
        let shared = history_geometry(18_432, 262_144, 1_024).unwrap();
        assert!(shared.span_limit <= MAX_HISTORY_SPANS);
        for (row_limit, appended) in [(1, 0), (65_792, 0), (262_144, 512), (1 << 20, 1_024)] {
            let geometry = history_geometry(4_096, row_limit, appended).unwrap();
            assert!(geometry.span_limit <= MAX_HISTORY_SPANS);
            if geometry.page_rows > SLAB_ROW_TILE {
                let smaller = history_geometry_span_limit(
                    geometry.page_rows - SLAB_ROW_TILE,
                    row_limit,
                    appended,
                );
                assert!(smaller > MAX_HISTORY_SPANS, "{row_limit} {appended}");
            }
        }
        assert_eq!(banks_per_slab(52 * 1024 * 1024).unwrap(), 1);
    }

    fn history_geometry_span_limit(page_rows: usize, row_limit: usize, appended: usize) -> usize {
        let appended = if appended == 0 {
            0
        } else {
            appended.div_ceil(page_rows) + 1
        };
        row_limit.div_ceil(page_rows) + 1 + appended
    }

    fn weight(shape: &[u64]) -> WeightDescriptor {
        WeightDescriptor::stored("w", shape)
    }

    fn norm(width: u64) -> RmsNorm {
        RmsNorm {
            weight: weight(&[width]),
            epsilon: 1e-6,
        }
    }

    fn block(op: Operator) -> Block {
        Block {
            sublayers: vec![Sublayer {
                input: InputNorm::Rms(norm(32)),
                op,
                output: OutputForm::Residual,
            }],
        }
    }

    fn attention() -> Operator {
        attention_in(KeyValue::Owned {
            key: weight(&[64, 32]),
            value: ValueSource::Projected(weight(&[64, 32])),
            key_norm: HeadNorm::None,
            value_norm: ValueNorm::None,
            domain: HistoryDomain::Token,
        })
    }

    fn window_attention(tokens: u64) -> Operator {
        attention_in(KeyValue::Owned {
            key: weight(&[64, 32]),
            value: ValueSource::Key,
            key_norm: HeadNorm::None,
            value_norm: ValueNorm::None,
            domain: HistoryDomain::Window { tokens },
        })
    }

    fn attention_in(key_value: KeyValue) -> Operator {
        let Operator::Attention(mut attention) = base_attention() else {
            unreachable!("an attention operator")
        };
        attention.key_value = key_value;
        Operator::Attention(attention)
    }

    fn base_attention() -> Operator {
        Operator::Attention(Box::new(Attention {
            heads: 4,
            kv_heads: 2,
            width: 32,
            query: weight(&[128, 32]),
            gate: AttentionGate::None,
            query_norm: HeadNorm::None,
            key_value: KeyValue::Owned {
                key: weight(&[64, 32]),
                value: ValueSource::Projected(weight(&[64, 32])),
                key_norm: HeadNorm::None,
                value_norm: ValueNorm::None,
                domain: HistoryDomain::Token,
            },
            rotary: Rotary::Interleaved {
                width: 8,
                base: 10_000.0,
                sections: vec![2, 1, 1],
                axis_pattern: vec![0, 1, 2],
            },
            scale: 1.0,
            reads: HistoryReads::Visible,
            media_rows: MediaRowAttention::Causal,
            sinks: None,
            output: weight(&[32, 128]),
        }))
    }

    fn decoder(activation_dtype: ActivationDType, blocks: Vec<Block>) -> Decoder {
        Decoder {
            activation_dtype,
            hidden: 32,
            vocabulary: 64,
            context_limit: 128,
            residual: ResidualForm::Single,
            entry: EntryForm {
                embedding: weight(&[64, 32]),
                scale: EmbeddingScale::Unit,
                norm: None,
                per_layer: None,
                hash_routing: None,
            },
            blocks,
            exit: ExitForm {
                norm: ExitNorm::Rms(norm(32)),
                output: weight(&[64, 32]),
                softcap: None,
            },
        }
    }

    #[test]
    fn derives_target_recurrent_and_dense_head_layouts() {
        let recurrent = GatedDelta {
            convolution_width: 4,
            key_heads: 2,
            value_heads: 4,
            width: 8,
            head_mapping: RecurrentHeadMapping::Grouped,
            query_key_value: weight(&[64, 32]),
            gate: weight(&[32, 32]),
            alpha: weight(&[4, 32]),
            beta: weight(&[4, 32]),
            convolution: weight(&[64, 4]),
            decay: weight(&[4]),
            time_bias: weight(&[4]),
            norm: norm(8),
            output_gate: GateFunction::Silu,
            output: weight(&[32, 32]),
        };
        let decoder = decoder(
            ActivationDType::BF16,
            vec![
                block(attention()),
                block(Operator::GatedDelta(Box::new(recurrent))),
            ],
        );
        let head = [block(attention()), block(attention())];
        let layout = ModelStateLayout::derive(
            &decoder,
            &head.iter().collect::<Vec<_>>(),
            KvCodec::AffineK8V4,
            3,
        )
        .unwrap();
        assert_eq!(layout.target_history.len(), 1);
        assert_eq!(
            layout.target_history[0].components()[0].layer,
            LayerRef::Target(0)
        );
        // 2 heads of width 32: 8-bit key code rows of 8 words and one affine
        // group (one (scale, zero) pair) per (row, kv head) vector.
        assert_eq!(layout.target_history[0].components()[0].codec.key_width, 32);
        assert_eq!(layout.target_history[0].components()[0].heads, 2);
        assert_eq!(
            layout.target_history[0].components()[0].planes()[0].row_extents,
            [2, 8]
        );
        assert_eq!(
            layout.target_history[0].components()[0].planes()[1].row_extents,
            [2, 2]
        );
        // Window: 3 history rows + 3 tape rows of 64 channels; delta; tape rows
        // of u [4, 8] | k [2, 8] | d [4].
        assert_eq!(layout.target_recurrent.len(), 3);
        assert_eq!(layout.target_recurrent[0].shape, [6, 64]);
        assert_eq!(layout.target_recurrent[0].dtype, DType::BF16);
        assert_eq!(layout.target_recurrent[1].shape, [4, 8, 8]);
        assert_eq!(layout.target_recurrent[1].dtype, DType::F32);
        assert_eq!(layout.target_recurrent[2].shape, [3, 52]);
        assert_eq!(layout.target_recurrent[2].dtype, DType::F32);
        assert_eq!(layout.head_history.len(), 1);
        assert_eq!(layout.head_history[0].components().len(), 2);
        assert_eq!(
            layout.head_history[0].components()[1].layer,
            LayerRef::Head(1)
        );
        assert_eq!(layout.head_history[0].components()[1].codec.key_width, 32);
        assert_eq!(
            layout.head_history[0].components()[1].planes()[0].row_extents,
            [2, 32]
        );
        assert!(matches!(
            layout.head_history[0].components()[1].codec.key,
            crate::Codec::Dense { dtype: DType::BF16 }
        ));
    }

    /// A block's bank list: its mixer's components, then its n-gram input's
    /// window of convolution inputs and its token window.
    #[test]
    fn an_ngram_input_banks_its_window_and_its_token_window_after_the_mixer() {
        let short = ShortConv {
            channels: 16,
            width: 3,
            input_gate: weight(&[16, 32]),
            value: weight(&[16, 32]),
            output_gate: weight(&[16, 32]),
            convolution: weight(&[16, 3]),
            output: weight(&[32, 16]),
        };
        let ngram = magnitude_family_contracts::NGramInput {
            hash: magnitude_family_contracts::NGramHash {
                context: 2,
                heads_per_order: 1,
                multipliers: vec![3, 5, 7],
                sizes: vec![5, 7],
                offsets: vec![0, 5],
                reset: 7,
            },
            width: 2,
            table: weight(&[12, 2]),
            key: weight(&[128, 4]),
            value: weight(&[32, 4]),
            key_norm: norm(128),
            query_norm: norm(128),
            convolution_norm: norm(128),
            convolution: weight(&[128, 4]),
            kernel: 4,
            dilation: 3,
        };
        let mut decoder = decoder(
            ActivationDType::BF16,
            vec![Block {
                sublayers: vec![
                    Sublayer {
                        input: InputNorm::Rms(norm(32)),
                        op: Operator::ShortConv(Box::new(short)),
                        output: OutputForm::Residual,
                    },
                    Sublayer {
                        input: InputNorm::None,
                        op: Operator::NGramInput(Box::new(ngram)),
                        output: OutputForm::Residual,
                    },
                ],
            }],
        );
        decoder.residual = ResidualForm::HyperConnections(
            magnitude_family_contracts::HyperConnections {
                streams: 4,
                mixing: magnitude_family_contracts::HyperConnectionMixing::LowRankGated { rank: 2 },
            },
        );
        let layout = ModelStateLayout::derive(&decoder, &[], KvCodec::Dense, 2).unwrap();
        assert_eq!(
            layout.target_banks,
            [
                BankComponent::ConvWindow {
                    width: 3,
                    channels: 16,
                    dtype: DType::F32,
                },
                BankComponent::ConvWindow {
                    width: 10,
                    channels: 128,
                    dtype: DType::F32,
                },
                BankComponent::ConvWindow {
                    width: 3,
                    channels: 1,
                    dtype: DType::U32,
                },
            ]
        );
        // Every window records the same 2 tape rows after its reach.
        assert_eq!(layout.target_recurrent[1].shape, [9 + 2, 128]);
        assert_eq!(layout.target_recurrent[2].shape, [2 + 2, 1]);
        assert_eq!(layout.target_recurrent[2].dtype, DType::U32);
    }

    #[test]
    fn dense_target_history_preserves_head_and_width_axes() {
        let decoder = decoder(ActivationDType::F16, vec![block(attention())]);
        let layout = ModelStateLayout::derive(&decoder, &[], KvCodec::Dense, 0).unwrap();
        assert_eq!(
            layout.target_history[0].components()[0].planes()[0].row_extents,
            [2, 32]
        );
        assert_eq!(
            layout.target_history[0].components()[0].planes()[1].row_extents,
            [2, 32]
        );
    }

    #[test]
    fn groups_history_domains_and_derives_every_bank_kind() {
        let short = ShortConv {
            channels: 16,
            width: 3,
            input_gate: weight(&[16, 32]),
            value: weight(&[16, 32]),
            output_gate: weight(&[16, 32]),
            convolution: weight(&[16, 3]),
            output: weight(&[32, 16]),
        };
        let space = StateSpace {
            heads: 4,
            head_width: 8,
            state: 16,
            groups: 2,
            convolution_width: 4,
            projection: weight(&[132, 32]),
            convolution: weight(&[96, 4]),
            convolution_bias: weight(&[96]),
            time_bias: weight(&[4]),
            decay: weight(&[4]),
            skip: weight(&[4]),
            norm: norm(32),
            norm_group: 16,
            output: weight(&[32, 32]),
        };
        let shared = |block| {
            attention_in(KeyValue::Shared {
                source: SublayerIndex { block, sublayer: 0 },
            })
        };
        let decoder = decoder(
            ActivationDType::BF16,
            vec![
                block(window_attention(4)),
                block(attention()),
                block(window_attention(8)),
                block(window_attention(4)),
                block(shared(0)),
                block(shared(1)),
                block(shared(0)),
                block(Operator::ShortConv(Box::new(short))),
                block(Operator::StateSpace(Box::new(space))),
            ],
        );
        let layout = ModelStateLayout::derive(&decoder, &[], KvCodec::Dense, 2).unwrap();
        let layers = |domain: &HistoryDomainLayout| {
            domain
                .components()
                .iter()
                .map(|component| component.layer)
                .collect::<Vec<_>>()
        };
        assert_eq!(layout.target_history.len(), 5);
        assert_eq!(
            layout.target_history[0].kind(),
            crate::HistoryDomainKind::Token
        );
        assert_eq!(layers(&layout.target_history[0]), [LayerRef::Target(1)]);
        assert_eq!(
            layout.target_history[1].kind(),
            crate::HistoryDomainKind::Window { rows: 4 }
        );
        assert_eq!(
            layers(&layout.target_history[1]),
            [LayerRef::Target(0), LayerRef::Target(3)]
        );
        assert_eq!(
            layout.target_history[2].kind(),
            crate::HistoryDomainKind::Window { rows: 8 }
        );
        assert_eq!(
            layout.target_history[3],
            HistoryDomainLayout::Shared {
                source: LayerRef::Target(0),
                layers: vec![LayerRef::Target(4), LayerRef::Target(6)],
            }
        );
        assert_eq!(
            layout.target_history[4],
            HistoryDomainLayout::Shared {
                source: LayerRef::Target(1),
                layers: vec![LayerRef::Target(5)],
            }
        );
        assert_eq!(
            layout.target_banks,
            [
                BankComponent::ConvWindow {
                    width: 3,
                    channels: 16,
                    dtype: DType::F32,
                },
                BankComponent::ConvWindow {
                    width: 4,
                    channels: 96,
                    dtype: DType::BF16,
                },
                BankComponent::SsmState {
                    heads: 4,
                    head_width: 8,
                    state_width: 16,
                },
                BankComponent::SsmTape {
                    heads: 4,
                    head_width: 8,
                    groups: 2,
                    state_width: 16,
                },
            ]
        );
        assert_eq!(layout.target_recurrent[0].shape, [2 + 2, 16]);
        assert_eq!(layout.target_recurrent[3].shape, [2, 32 + 32 + 4]);
        assert!(layout.head_history.is_empty());
    }
}
