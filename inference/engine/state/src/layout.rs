use crate::{
    BankComponent, CodecSpec, ComponentDescriptor, ComponentSpec, HistoryDomainLayout, KvCodec,
    LayerRef,
};
use magnitude_family_contracts::{
    ActivationDType, Attention, Block, Decoder, GatedDelta, HistoryReads, HistoryRows, Indexer,
    IndexerKeys, NGramInput, Operator, Pool, PooledRows, ProjectedRows, ResidualForm,
    RowProducer, RowRounding, Selection, ShortConv, StateSpace, TokenDomain, ValueSource,
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

/// The free-page rows a history takes to append `count` rows when its last
/// page holds `in_place` more in place: whatever the last page cannot hold,
/// rounded up to whole pages of `page_rows`. The one rule of a history's
/// page demand (`StateStore::page_demand`, [`crate::batch_state_bytes`]).
pub(crate) fn page_demand_rows(page_rows: usize, in_place: usize, count: usize) -> usize {
    count.saturating_sub(in_place).div_ceil(page_rows) * page_rows
}

/// The most pages a Window history occupies, its histories holding at most
/// `row_limit` rows at once ([`crate::HistoryDomainKind::row_limit`]): it
/// releases its rows before its window as it slides, so it holds its row
/// limit's pages and the partial page at its front that the slide leaves,
/// however long it grows. Plan, assessment and the ledger's claims size a
/// window history by it.
pub fn window_pages(page_rows: usize, row_limit: usize) -> usize {
    row_limit.div_ceil(page_rows) + 1
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
/// in that order (each only when present), then the produced domains, per
/// distinct rate in order of first use: a Window of `rate − 1` rows (above
/// rate 1) holding pooled sources' incomplete groups and pooled indexer
/// keys' raw projections, and a Block of that rate holding pooled sources'
/// rows and indexer keys. Recurrent banks list each
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
            for stateful in block_state(block)? {
                match stateful {
                    Stateful::Attention(attention) => target_history.attention(
                        layer(index)?,
                        attention,
                        target_codec,
                        activation,
                        Owner::Target,
                    )?,
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
                // drafters' attention reads dense planes only. A drafter
                // keeps every source's own state, its indexer's too.
                [Stateful::Attention(attention)] => {
                    if attention
                        .history
                        .iter()
                        .any(|source| matches!(source.rows, HistoryRows::Shared { .. }))
                    {
                        return Err("a drafter block must own its attention history".into());
                    }
                    head_history.attention(
                        layer(index)?,
                        attention,
                        KvCodec::Dense,
                        activation,
                        Owner::Head,
                    )?
                }
                _ => {
                    return Err("a drafter block must hold exactly one owned attention history".into());
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
    /// The produced domains of each rate, in order of first use: the
    /// incomplete groups' components (a Window of `rate − 1` rows, empty at
    /// rate 1) and the rows' and keys' (a Block of that rate).
    produced: Vec<Produced>,
}

/// The produced components of one rate.
struct Produced {
    rate: usize,
    window: Vec<ComponentDescriptor>,
    block: Vec<ComponentDescriptor>,
}

impl HistoryDomains {
    /// Target block `block`'s attention: each source's state, in source
    /// order. Part 0 is projected (validation keeps a pooled source out of
    /// the first place): its rows are `Target(block)`'s, under `codec`
    /// unless the model rounds them. A later part is pooled: its rows are
    /// `Pooled { block, part }`'s (`HeadPooled` in a head's). A Shared source's rows, and an indexer's
    /// Shared keys, are their owner's. A source's own indexer keeping its
    /// own keys keeps them as `IndexKeys { block, part }` (`HeadIndexKeys`).
    fn attention(
        &mut self,
        block: u32,
        attention: &Attention,
        codec: KvCodec,
        activation: DType,
        owner: Owner,
    ) -> Result<(), String> {
        for (part, source) in attention.history.iter().enumerate() {
            let part = u32::try_from(part).map_err(|_| "history source index".to_string())?;
            let layer = owner.source(block, part);
            match &source.rows {
                HistoryRows::Own(RowProducer::Projected(rows)) => {
                    self.own(rows.domain, history_component(attention, layer, codec, activation)?)?
                }
                HistoryRows::Own(RowProducer::Pooled(rows)) => {
                    self.pool(layer, rows, attention.width, activation)?
                }
                HistoryRows::Shared {
                    source,
                    part: source_part,
                } => {
                    let source_part = u32::try_from(*source_part)
                        .map_err(|_| "history source index".to_string())?;
                    self.share(owner.source(source.block, source_part), layer);
                }
            }
            match &source.reads {
                HistoryReads::Selected(Selection::Own(indexer))
                    if !matches!(indexer.keys, IndexerKeys::Shared { .. }) =>
                {
                    self.index(owner.keys(block, part), indexer, &source.rows)?
                }
                HistoryReads::Visible | HistoryReads::Selected(_) => {}
            }
        }
        Ok(())
    }

    /// A pooled source's state: at rate > 1 each pending token's F32
    /// projection (and, gated, its F32 gate scores beside it) with its
    /// coordinates, in the rate's Window; its rows, one `width` head each,
    /// in the rate's Block, in BF16 when the model rounds them (every
    /// rounding's values are exact in BF16) and in the activation dtype
    /// otherwise.
    fn pool(
        &mut self,
        layer: LayerRef,
        rows: &PooledRows,
        width: u64,
        activation: DType,
    ) -> Result<(), String> {
        let rate = host(rows.rate, "pooled source rate")?;
        let width = host(width, "pooled source width")?;
        let window = pending(layer, &rows.pool, width)?;
        let dtype = match rows.rounding {
            RowRounding::None => activation,
            RowRounding::Block(_) => DType::BF16,
        };
        let block = ComponentDescriptor::model_history(
            layer,
            CodecSpec::stored_key(dtype, width),
            1,
        )
        .map_err(|error| error.to_string())?;
        self.produce(rate, window, block)
    }

    /// The keys of an indexer selecting `rows`:
    /// - pooled keys over a token source (Qwen4's, GLM's): each pending
    ///   token's F32 key row (gated, with its gate scores beside it) with
    ///   its coordinates in the rate's Window, as a pooled source's
    ///   ([`pending`]), and each group's key of `width` F32 values in the
    ///   rate's Block (the indexer scores in F32, as the reference does);
    /// - keys of the layer's pooled source's rows: one row per source row,
    ///   in the source's Block, so its rows are the source's, in F32.
    fn index(
        &mut self,
        keys: LayerRef,
        indexer: &Indexer,
        rows: &HistoryRows,
    ) -> Result<(), String> {
        let width = host(indexer.width, "indexer width")?;
        let component = |dtype, coordinates| {
            ComponentDescriptor::produced(keys, dtype, width, coordinates)
                .map_err(|error| error.to_string())
        };
        match (&indexer.keys, rows) {
            // A rounding would be state no indexer form keeps.
            (
                IndexerKeys::Pooled(
                    pooled @ PooledRows {
                        rounding: RowRounding::None,
                        ..
                    },
                ),
                _,
            ) => {
                let rate = host(pooled.rate, "selection block")?;
                if rate < 2 {
                    return Err("a selection block holds at least two rows".into());
                }
                let window = pending(keys, &pooled.pool, width)?
                    .ok_or("pooled indexer keys of a pool without pending tokens")?;
                self.produce(rate, Some(window), component(DType::F32, false)?)
            }
            (IndexerKeys::Pooled(_), _) => {
                Err("rounded pooled indexer keys have no state layout".into())
            }
            // F32 holds a rounded key's values exactly, and the scoring
            // entry reads one key element (`indexer_score`).
            (IndexerKeys::Latent { .. }, HistoryRows::Own(RowProducer::Pooled(pooled))) => {
                let rate = host(pooled.rate, "pooled source rate")?;
                self.produce(rate, None, component(DType::F32, false)?)
            }
            (IndexerKeys::Latent { .. }, _) => {
                Err("latent indexer keys over rows the layer does not pool".into())
            }
            (IndexerKeys::Shared { .. }, _) => Err("shared indexer keys are their owner's state".into()),
        }
    }

    fn produce(
        &mut self,
        rate: usize,
        window: Option<ComponentDescriptor>,
        block: ComponentDescriptor,
    ) -> Result<(), String> {
        if rate == 0 || (rate == 1 && window.is_some()) {
            return Err("produced rows of an invalid rate".into());
        }
        let produced = match self.produced.iter_mut().position(|produced| produced.rate == rate) {
            Some(index) => &mut self.produced[index],
            None => {
                self.produced.push(Produced {
                    rate,
                    window: Vec::new(),
                    block: Vec::new(),
                });
                self.produced.last_mut().expect("just pushed")
            }
        };
        produced.window.extend(window);
        produced.block.push(block);
        Ok(())
    }

    fn own(&mut self, domain: TokenDomain, component: ComponentDescriptor) -> Result<(), String> {
        match domain {
            TokenDomain::Token => self.token.push(component),
            TokenDomain::Window { tokens } => {
                let rows = host(tokens, "attention window")?;
                match self.windows.iter_mut().find(|(window, _)| *window == rows) {
                    Some((_, components)) => components.push(component),
                    None => self.windows.push((rows, vec![component])),
                }
            }
        }
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
            .chain(self.produced.into_iter().flat_map(|produced| {
                let window = (!produced.window.is_empty()).then(|| HistoryDomainLayout::Window {
                    rows: produced.rate - 1,
                    components: produced.window,
                });
                window.into_iter().chain([HistoryDomainLayout::Block {
                    rate: produced.rate,
                    components: produced.block,
                }])
            }))
            .collect()
    }
}

/// The state one stateful sublayer keeps.
enum Stateful<'a> {
    Attention(&'a Attention),
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
            Operator::Attention(attention) => Stateful::Attention(attention),
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
            op @ Operator::Parallel(_) => {
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

/// The component of an attention layer's first source, its own projected
/// rows: in the host's policy, or the model's own store ([`model_store`]).
fn history_component(
    attention: &Attention,
    layer: LayerRef,
    codec: KvCodec,
    activation: DType,
) -> Result<ComponentDescriptor, String> {
    let kv_heads = host(attention.kv_heads, "attention KV heads")?;
    let spec = history_codec(attention, codec, activation)?;
    let component = if model_store(attention) {
        ComponentDescriptor::model_history(layer, spec, kv_heads)
    } else {
        ComponentDescriptor::new(layer, spec, kv_heads)
    };
    component.map_err(|error| error.to_string())
}

/// Whether an attention layer's first source holds the model's own dense
/// store whatever the host policy: rows the model rounds
/// (`RowRounding::Block`, stored-key by validation), stored-key rows
/// (`ValueSource::StoredKey`; stored keys are dense only), and a layer's
/// rows beside an index-list source, which only the dense entries read.
/// The codec ([`history_codec`]), the component's role and the policy's
/// refusal of a lossy codec ([`kv_codec_policy`]) all follow it.
fn model_store(attention: &Attention) -> bool {
    attention.history.len() > 1
        || projected_rows(attention).is_some_and(|rows| {
            rows.rounding != RowRounding::None || rows.value == ValueSource::StoredKey
        })
}

/// The codec of the rows an attention layer's first source holds (the
/// source its fresh rows join), under the host's storage policy `codec`
/// with dense planes in `activation`: per head, its key and its value's own
/// vector, or for stored-key values (`ValueSource::StoredKey`) the dense key
/// alone. The model's own store ([`model_store`]) is dense whatever
/// `codec` ([`kv_codec_policy`] refuses a lossy policy named for it): rows
/// the model rounds in BF16, which holds every rounded value exactly, the
/// others in `activation`. `codec` governs the other unrounded per-token
/// rows alone; pooled sources' rows keep their own store too
/// (`ModelStateLayout::derive`). State allocation and the attention entries
/// both take it from here. A Shared layer reads its source's rows, of its
/// width; the executor admits no Shared reader of stored keys
/// (`operators::attention::admit_shared`).
pub fn history_codec(
    attention: &Attention,
    codec: KvCodec,
    activation: DType,
) -> Result<CodecSpec, String> {
    let width = host(attention.width, "attention head width")?;
    let codec = if model_store(attention) { KvCodec::Dense } else { codec };
    let spec = match projected_rows(attention) {
        Some(ProjectedRows {
            rounding: RowRounding::Block(_),
            ..
        }) => CodecSpec::stored_key(DType::BF16, width),
        Some(ProjectedRows {
            value: ValueSource::StoredKey,
            ..
        }) => CodecSpec::stored_key(activation, width),
        Some(_) | None => codec.spec(activation, width, width),
    };
    spec.validate().map_err(|error| error.to_string())
}

/// The host's KV policy for `decoder`'s unrounded per-token history:
/// `requested` when named, else the serving default ([`KvCodec::AffineK8V4`]).
/// The model's own store ([`model_store`]) stays dense under either
/// ([`history_codec`]); a lossy policy named for a model with such rows is
/// refused, since it can't apply to them and is never silently replaced,
/// so such a model's default is the one policy it admits, dense: the policy
/// its store identifies as (`advance::identify_codec`).
pub fn kv_codec_policy(decoder: &Decoder, requested: Option<KvCodec>) -> Result<KvCodec, String> {
    let own_store = decoder.blocks.iter().flat_map(|block| &block.sublayers).any(|sublayer| {
        matches!(&sublayer.op, Operator::Attention(attention) if model_store(attention))
    });
    match requested {
        None if own_store => Ok(KvCodec::Dense),
        None => Ok(KvCodec::AffineK8V4),
        Some(codec) if own_store && codec != KvCodec::Dense => {
            Err("a lossy KV codec over the model's dense-only history rows".into())
        }
        Some(codec) => Ok(codec),
    }
}

/// The Window component of `layer`'s pending tokens under `pool` over rows
/// of `width` (`pool_rows`' `window_rows`): each token's F32 row, gated with
/// its F32 gate scores beside it, and its coordinates; none at rate 1.
fn pending(layer: LayerRef, pool: &Pool, width: usize) -> Result<Option<ComponentDescriptor>, String> {
    let pending = match pool {
        Pool::Identity => return Ok(None),
        Pool::Mean => width,
        Pool::SoftmaxGated { .. } => width
            .checked_mul(2)
            .ok_or("pooled source width arithmetic overflow")?,
    };
    ComponentDescriptor::produced(layer, DType::F32, pending, true)
        .map(Some)
        .map_err(|error| error.to_string())
}

/// Whose store an attention block's state is kept in: a target block's, or
/// a draft head block's (whose `block` is its index among the head's blocks).
#[derive(Clone, Copy)]
enum Owner {
    Target,
    Head,
}

impl Owner {
    /// The state layer of the block's history source `part`: its token rows
    /// (part 0) or its pooled rows.
    fn source(self, block: u32, part: u32) -> LayerRef {
        match (self, part) {
            (Self::Target, 0) => LayerRef::Target(block),
            (Self::Head, 0) => LayerRef::Head(block),
            (Self::Target, part) => LayerRef::Pooled { block, part },
            (Self::Head, part) => LayerRef::HeadPooled { block, part },
        }
    }

    /// The state layer of the keys of the indexer selecting source `part`.
    fn keys(self, block: u32, part: u32) -> LayerRef {
        match self {
            Self::Target => LayerRef::IndexKeys { block, part },
            Self::Head => LayerRef::HeadIndexKeys { block, part },
        }
    }
}

/// An attention layer's own projected rows (its first source's), if it
/// owns them.
fn projected_rows(attention: &Attention) -> Option<&ProjectedRows> {
    match &attention.fresh().rows {
        HistoryRows::Own(RowProducer::Projected(rows)) => Some(rows),
        HistoryRows::Own(RowProducer::Pooled(_)) | HistoryRows::Shared { .. } => None,
    }
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
            decay_width: host(recurrent.decay_width(), "recurrent decay width")?,
        },
    ])
}

/// The n-gram input's bank components, in the order its entries take them:
/// with a convolution, its window of u rows (`ngram_inject`), F32, over the
/// decoder's S * H stream channels, `dilation (kernel - 1)` rows before the
/// state (a dense window of that reach plus one); then its token window
/// (`ngram_rows`), the sequence's `context` tokens before the state, each
/// stored as token + 1 in one U32 channel so the zero seed holds no token.
fn ngram_bank(decoder: &Decoder, input: &NGramInput) -> Result<Vec<BankComponent>, String> {
    let ResidualForm::HyperConnections(streams) = &decoder.residual else {
        return Err("an n-gram input without residual streams".into());
    };
    let mut bank = Vec::with_capacity(2);
    if let Some(mixing) = &input.mixing {
        let reach = mixing
            .dilation
            .checked_mul(
                mixing
                    .kernel
                    .checked_sub(1)
                    .ok_or("n-gram kernel must be positive")?,
            )
            .ok_or("n-gram reach overflows")?;
        bank.push(BankComponent::ConvWindow {
            width: host(reach + 1, "n-gram convolution reach")?,
            channels: host(
                streams
                    .streams
                    .checked_mul(decoder.hidden)
                    .ok_or("n-gram channels overflow")?,
                "n-gram channels",
            )?,
            dtype: DType::F32,
        });
    }
    bank.push(BankComponent::ConvWindow {
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
    });
    Ok(bank)
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
    use crate::ComponentRole;
    use magnitude_family_contracts::{
        AttentionGate, BlockElement, BlockRounding, BlockScale, EmbeddingScale, EntryForm,
        ExitForm, ExitNorm, GateFunction, GateProjection, RecurrentDecay, RecurrentGate, HeadNorm, HeadWeights, IndexerInput, IndexerQuery,
        SublayerIndex, HistorySource, InputNorm, MediaRowAttention, OutputForm, OutputProjection, QueryKeys, QueryProjection,
        PoolNorm, RecurrentHeadMapping, ResidualForm, RmsNorm, Rotary, Sublayer, ValueNorm, ValueSource,
        WeightDescriptor,
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
        attention_in(owned(
            ValueSource::Projected(weight(&[64, 32])),
            TokenDomain::Token,
        ))
    }

    fn window_attention(tokens: u64) -> Operator {
        attention_in(owned(ValueSource::RawKey, TokenDomain::Window { tokens }))
    }

    /// Own projected rows of two 32-wide kv heads, unnormed and unrounded.
    fn owned(value: ValueSource, domain: TokenDomain) -> HistoryRows {
        HistoryRows::Own(RowProducer::Projected(ProjectedRows {
            key: weight(&[64, 32]),
            value,
            key_norm: HeadNorm::None,
            value_norm: ValueNorm::None,
            domain,
            rounding: RowRounding::None,
        }))
    }

    fn attention_in(rows: HistoryRows) -> Operator {
        let Operator::Attention(mut attention) = base_attention() else {
            unreachable!("an attention operator")
        };
        attention.history[0].rows = rows;
        Operator::Attention(attention)
    }

    fn base_attention() -> Operator {
        Operator::Attention(Box::new(Attention {
            heads: 4,
            kv_heads: 2,
            width: 32,
            query: QueryProjection::Full(weight(&[128, 32])),
            query_keys: QueryKeys::Direct,
            gate: AttentionGate::None,
            query_norm: HeadNorm::None,
            history: vec![HistorySource::visible(owned(
                ValueSource::Projected(weight(&[64, 32])),
                TokenDomain::Token,
            ))],
            rotary: Rotary::Interleaved {
                width: 8,
                base: 10_000.0,
                sections: vec![2, 1, 1],
                axis_pattern: vec![0, 1, 2],
            },
            scale: 1.0,
            media_rows: MediaRowAttention::Causal,
            sinks: None,
            output: OutputProjection::Full(weight(&[32, 128])),
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
            beta: weight(&[4, 32]),
            convolution: weight(&[64, 4]),
            decay: RecurrentDecay::PerHead {
                alpha: weight(&[4, 32]),
                time_bias: weight(&[4]),
                decay: weight(&[4]),
            },
            key_norm_epsilon: 8e-6,
            norm: norm(8),
            output_gate: RecurrentGate {
                projection: GateProjection::Full(weight(&[32, 32])),
                function: GateFunction::Silu,
            },
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

    /// A window layer whose values are its stored keys holds the key plane
    /// alone, half the row bytes of the same layer over K and V; the raw-key
    /// layer beside it keeps both planes.
    #[test]
    fn a_stored_key_layer_holds_one_plane() {
        let stored = attention_in(owned(
            ValueSource::StoredKey,
            TokenDomain::Window { tokens: 16 },
        ));
        let decoder = decoder(
            ActivationDType::BF16,
            vec![block(stored), block(window_attention(16))],
        );
        let layout = ModelStateLayout::derive(&decoder, &[], KvCodec::Dense, 0).unwrap();
        let [domain] = &layout.target_history[..] else {
            panic!("one window domain");
        };
        let [stored, raw] = domain.components() else {
            panic!("two layers");
        };
        assert_eq!(
            stored
                .planes()
                .iter()
                .map(|plane| (plane.vector, plane.row_extents.clone()))
                .collect::<Vec<_>>(),
            [(crate::VectorKind::Key, vec![2, 32])]
        );
        assert_eq!(raw.planes().len(), 2);
        assert_eq!(2 * stored.row_bytes().unwrap(), raw.row_bytes().unwrap());
        let Operator::Attention(attention) = &decoder.blocks[0].sublayers[0].op else {
            unreachable!()
        };
        assert_eq!(
            history_codec(attention, KvCodec::Dense, DType::BF16).unwrap(),
            stored.codec
        );
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
                token_map: None,
                fill: 7,
                boundary: Some(7),
                media: magnitude_family_contracts::NGramMedia::Hashed,
            },
            width: 2,
            table: weight(&[12, 2]),
            key: weight(&[128, 4]),
            value: weight(&[32, 4]),
            gate: magnitude_family_contracts::NGramGate::Norms {
                key: weight(&[128]),
                query: weight(&[128]),
            },
            epsilon: 1e-6,
            mixing: Some(magnitude_family_contracts::NGramConvolution {
                norm: weight(&[128]),
                taps: weight(&[128, 4]),
                kernel: 4,
                dilation: 3,
            }),
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
        // Without a convolution, its token window alone.
        let Operator::NGramInput(input) = &mut decoder.blocks[0].sublayers[1].op else {
            unreachable!()
        };
        input.mixing = None;
        let layout = ModelStateLayout::derive(&decoder, &[], KvCodec::Dense, 2).unwrap();
        assert_eq!(layout.target_banks.len(), 2);
        assert_eq!(layout.target_recurrent[1].shape, [2 + 2, 1]);
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
            attention_in(HistoryRows::Shared {
                source: SublayerIndex { block, sublayer: 0 },
                part: 0,
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

    const V41_WIDTH: u64 = 512;

    fn block_rounding(element: BlockElement, block: u64, scale: BlockScale, floor: f64) -> RowRounding {
        RowRounding::Block(BlockRounding {
            element,
            block,
            scale,
            amax_floor: floor,
        })
    }

    /// V4.1's 128-row window of one 512-wide stored-key head, E4M3 per 32.
    fn v41_window() -> HistorySource {
        HistorySource::visible(HistoryRows::Own(RowProducer::Projected(ProjectedRows {
            key: weight(&[V41_WIDTH, 32]),
            value: ValueSource::StoredKey,
            key_norm: HeadNorm::Rms(norm(V41_WIDTH)),
            value_norm: ValueNorm::None,
            domain: TokenDomain::Window { tokens: 128 },
            rounding: block_rounding(BlockElement::E4M3, 32, BlockScale::Ue8m0, 1e-4),
        })))
    }

    /// A V4.1 attention over `history`, one kv head 512 wide.
    fn v41_attention(history: Vec<HistorySource>) -> Operator {
        let Operator::Attention(mut attention) = base_attention() else {
            unreachable!("an attention operator")
        };
        attention.kv_heads = 1;
        attention.width = V41_WIDTH;
        attention.history = history;
        Operator::Attention(attention)
    }

    /// Slice C: two window-only layers, layer 2 Full (its window, a rate-2
    /// softmax-gated source selected by its own indexer over latent keys),
    /// layer 3 Reuse (its window, layer 2's source and picks).
    fn v41_slice_c() -> Decoder {
        let full = v41_attention(vec![
            v41_window(),
            HistorySource {
                rows: HistoryRows::Own(RowProducer::Pooled(PooledRows {
                    rate: 2,
                    projection: weight(&[V41_WIDTH, 32]),
                    pool: Pool::SoftmaxGated {
                        gate: weight(&[V41_WIDTH, 32]),
                        position: None,
                    },
                    norm: PoolNorm::Pooled(norm(V41_WIDTH)),
                    rounding: block_rounding(BlockElement::E2M1, 16, BlockScale::E4M3, 6.0 * 2f64.powi(-9)),
                })),
                reads: HistoryReads::Selected(Selection::Own(Box::new(Indexer {
                    heads: 32,
                    width: 128,
                    query: IndexerQuery {
                        input: IndexerInput::QueryRank,
                        projection: weight(&[32 * 128, 1280]),
                        norm: HeadNorm::None,
                        rounding: block_rounding(BlockElement::E2M1, 32, BlockScale::Ue8m0, 6.0 * 2f64.powi(-126)),
                    },
                    keys: IndexerKeys::Latent {
                        projection: weight(&[128, V41_WIDTH]),
                        norm: norm(128),
                        rounding: block_rounding(BlockElement::E2M1, 32, BlockScale::Ue8m0, 6.0 * 2f64.powi(-126)),
                    },
                    head_weights: HeadWeights::Projected(weight(&[32, 32])),
                    scale: 1.0 / (128.0f64 * 32.0).sqrt(),
                    top: 512,
                    candidates: magnitude_family_contracts::Candidates::All,
                }))),
            },
        ]);
        let source = SublayerIndex {
            block: 2,
            sublayer: 0,
        };
        let reuse = v41_attention(vec![
            v41_window(),
            HistorySource {
                rows: HistoryRows::Shared { source, part: 1 },
                reads: HistoryReads::Selected(Selection::Reuse { source, part: 1 }),
            },
        ]);
        let window = || v41_attention(vec![v41_window()]);
        decoder(
            ActivationDType::BF16,
            vec![block(window()), block(window()), block(full), block(reuse)],
        )
    }

    fn plane_bytes(component: &ComponentDescriptor) -> Vec<(crate::VectorKind, DType, usize)> {
        component
            .planes()
            .iter()
            .map(|plane| (plane.vector, plane.dtype, plane.row_bytes))
            .collect()
    }

    /// Slice C under the serving default (dense, the one policy its own
    /// stores admit): the windows and the pooled rows are the model's own
    /// BF16 stores whatever the host policy; the pooled
    /// source's incomplete group is a Window(1) of F32 projections and gate
    /// scores with coordinates; its rows and its indexer's keys share one
    /// Block(2), so their rows coincide; layer 3 shares part 1.
    #[test]
    fn v41_slice_c_lays_out_its_history_sources() {
        use crate::VectorKind::{Key, Value};
        let decoder = v41_slice_c();
        let codec = kv_codec_policy(&decoder, None).unwrap();
        assert_eq!(codec, KvCodec::Dense);
        let layout = ModelStateLayout::derive(&decoder, &[], codec, 0).unwrap();
        let pooled = LayerRef::Pooled { block: 2, part: 1 };
        let keys = LayerRef::IndexKeys { block: 2, part: 1 };
        let [
            HistoryDomainLayout::Window { rows: 128, components: windows },
            HistoryDomainLayout::Shared { source, layers },
            HistoryDomainLayout::Window { rows: 1, components: pending },
            HistoryDomainLayout::Block { rate: 2, components: rows },
        ] = &layout.target_history[..]
        else {
            panic!("window, shared, pending and block domains: {:?}", layout.target_history);
        };
        assert_eq!(
            windows.iter().map(|component| component.layer).collect::<Vec<_>>(),
            (0..4).map(LayerRef::Target).collect::<Vec<_>>()
        );
        for window in windows {
            assert_eq!(window.role, ComponentRole::ModelHistory);
            assert_eq!(plane_bytes(window), [(Key, DType::BF16, 1024)]);
        }
        assert_eq!((*source, layers.as_slice()), (pooled, &[LayerRef::Pooled { block: 3, part: 1 }][..]));
        let [pending] = &pending[..] else { panic!("one pending group") };
        assert_eq!((pending.layer, pending.role), (pooled, ComponentRole::Produced));
        assert_eq!(plane_bytes(pending), [(Key, DType::F32, 4096), (Value, DType::I32, 16)]);
        let [pooled_rows, index_keys] = &rows[..] else { panic!("rows and keys") };
        assert_eq!((pooled_rows.layer, pooled_rows.role), (pooled, ComponentRole::ModelHistory));
        assert_eq!(plane_bytes(pooled_rows), [(Key, DType::BF16, 1024)]);
        assert_eq!((index_keys.layer, index_keys.role), (keys, ComponentRole::Produced));
        assert_eq!(plane_bytes(index_keys), [(Key, DType::F32, 512)]);
        // The domains validate as one store: layer 3 reads layer 2's rows.
        let plans = layout
            .target_history
            .iter()
            .map(|layout| crate::HistoryDomainPlan {
                logical_rows: layout.kind().row_limit(128, 8).unwrap(),
                layout: layout.clone(),
            })
            .collect::<Vec<_>>();
        let reads = crate::HistoryReads::of(&plans, 128, 8).unwrap();
        // Stored: the window (0), the pending groups (1), the pooled rows
        // (2); then the shared read of the pooled rows.
        assert_eq!(reads.read(pooled), Some(2));
        assert_eq!(reads.read(LayerRef::Pooled { block: 3, part: 1 }), Some(3));
        assert_eq!(reads.read(keys), None);
    }

    /// A lossy policy can't apply to rows the model rounds: named, it is
    /// refused; unnamed, the default is dense.
    #[test]
    fn a_lossy_policy_named_for_rounded_rows_is_refused() {
        let rounded = v41_slice_c();
        for lossy in [KvCodec::AffineK8V4, KvCodec::RotatedK4V4] {
            assert_eq!(
                kv_codec_policy(&rounded, Some(lossy)),
                Err("a lossy KV codec over the model's dense-only history rows".into())
            );
        }
        assert_eq!(kv_codec_policy(&rounded, Some(KvCodec::Dense)), Ok(KvCodec::Dense));
        assert_eq!(kv_codec_policy(&rounded, None), Ok(KvCodec::Dense));
        let plain = decoder(ActivationDType::BF16, vec![block(attention())]);
        assert_eq!(
            kv_codec_policy(&plain, Some(KvCodec::RotatedK4V4)),
            Ok(KvCodec::RotatedK4V4)
        );
        assert_eq!(kv_codec_policy(&plain, None), Ok(KvCodec::AffineK8V4));
    }

    /// An attention whose selected read's indexer pools its keys by `pool`
    /// under `norm` at rate 4 over keys of 64.
    fn selected_attention(pool: Pool, norm: PoolNorm) -> Operator {
        let Operator::Attention(mut attention) = attention() else {
            unreachable!("an attention operator")
        };
        attention.history[0].reads = HistoryReads::Selected(Selection::Own(Box::new(Indexer {
            heads: 2,
            width: 64,
            query: IndexerQuery {
                input: IndexerInput::Hidden,
                projection: weight(&[128, 32]),
                norm: HeadNorm::Rms(self::norm(64)),
                rounding: RowRounding::None,
            },
            keys: IndexerKeys::Pooled(PooledRows {
                rate: 4,
                projection: weight(&[64, 32]),
                pool,
                norm,
                rounding: RowRounding::None,
            }),
            head_weights: HeadWeights::Uniform,
            scale: 0.125,
            top: 8,
            candidates: magnitude_family_contracts::Candidates::All,
        })));
        Operator::Attention(attention)
    }

    /// GLM's token-normed keys gated by place.
    fn gated_selected_attention() -> Operator {
        selected_attention(
            Pool::SoftmaxGated { gate: weight(&[64, 32]), position: Some(weight(&[4, 64])) },
            PoolNorm::Token {
                norm: magnitude_family_contracts::LayerNorm { weight: weight(&[64]), epsilon: 1e-6 },
                bias: weight(&[64]),
            },
        )
    }

    /// The Window and Block components of a selected read whose indexer
    /// pools its keys by `pool` under `norm` at rate 4 over keys of 64.
    fn pooled_key_layout(pool: Pool, norm: PoolNorm) -> (Vec<ComponentDescriptor>, Vec<ComponentDescriptor>) {
        let attention = selected_attention(pool, norm);
        let decoder = decoder(ActivationDType::BF16, vec![block(attention)]);
        let layout = ModelStateLayout::derive(&decoder, &[], KvCodec::AffineK8V4, 0).unwrap();
        let [
            HistoryDomainLayout::Token { components: token },
            HistoryDomainLayout::Window { rows: 3, components: raw },
            HistoryDomainLayout::Block { rate: 4, components: finished },
        ] = &layout.target_history[..]
        else {
            panic!("token, raw and block domains: {:?}", layout.target_history);
        };
        assert_eq!(token[0].role, ComponentRole::History);
        (raw.clone(), finished.clone())
    }

    fn index_keys(width: usize, coordinates: bool) -> ComponentDescriptor {
        keys_of(LayerRef::IndexKeys { block: 0, part: 0 }, width, coordinates)
    }

    fn keys_of(layer: LayerRef, width: usize, coordinates: bool) -> ComponentDescriptor {
        ComponentDescriptor::produced(layer, DType::F32, width, coordinates).unwrap()
    }

    /// Qwen4's selected read: raw F32 keys with coordinates in a Window of
    /// the block less one, F32 block keys in a Block of the block, after
    /// the attended domain: the layout before history sources, its keys now
    /// named by part.
    #[test]
    fn a_pooled_key_indexer_keeps_its_raw_and_block_keys() {
        let layout = pooled_key_layout(Pool::Mean, PoolNorm::Pooled(norm(64)));
        assert_eq!(layout, (vec![index_keys(64, true)], vec![index_keys(64, false)]));
    }

    /// GLM's: each pending token's normed key and its gate scores beside
    /// it (2W, as a gated pooled source's Window), the group keys in F32.
    #[test]
    fn a_gated_pooled_key_indexer_keeps_its_keys_and_gate_scores() {
        let layout = pooled_key_layout(
            Pool::SoftmaxGated { gate: weight(&[64, 32]), position: Some(weight(&[4, 64])) },
            PoolNorm::Token {
                norm: magnitude_family_contracts::LayerNorm { weight: weight(&[64]), epsilon: 1e-6 },
                bias: weight(&[64]),
            },
        );
        assert_eq!(layout, (vec![index_keys(128, true)], vec![index_keys(64, false)]));
    }

    /// A draft head's block selecting through its own indexer keeps the keys
    /// beside its rows, named by the head's block: the pending tokens' Window
    /// and the units' Block, after its Token domain.
    #[test]
    fn a_head_keeps_its_indexers_keys_beside_its_rows() {
        let decoder = decoder(ActivationDType::BF16, vec![block(attention())]);
        let head = [block(gated_selected_attention())];
        let layout = ModelStateLayout::derive(&decoder, &head.iter().collect::<Vec<_>>(), KvCodec::AffineK8V4, 0).unwrap();
        let [
            HistoryDomainLayout::Token { components: rows },
            HistoryDomainLayout::Window { rows: 3, components: pending },
            HistoryDomainLayout::Block { rate: 4, components: units },
        ] = &layout.head_history[..]
        else {
            panic!("token, pending and unit domains: {:?}", layout.head_history);
        };
        assert_eq!((rows[0].layer, rows[0].role), (LayerRef::Head(0), ComponentRole::History));
        assert!(matches!(rows[0].codec.key, crate::Codec::Dense { .. }), "a head's rows stay dense");
        let keys = LayerRef::HeadIndexKeys { block: 0, part: 0 };
        assert_eq!(pending, &vec![keys_of(keys, 128, true)]);
        assert_eq!(units, &vec![keys_of(keys, 64, false)]);
        // The target keeps none of them.
        assert_eq!(layout.target_history.len(), 1);
    }

    /// A drafting round's speculative rows complete units of the head's
    /// indexer keys beside its rows; accepting only the entry rows leaves the
    /// state a launch of them alone leaves, the unit and the pending keys the
    /// rejected rows wrote released, so the next round reads exactly what a
    /// run that stopped there would.
    #[test]
    fn a_rejected_draft_leaves_the_heads_pooled_rows_as_its_entry_rows_alone_do() {
        use crate::{BankCapacity, HistoryDomainPlan, OwnedAdvanceResolution, OwnedStateAdvance, StateStore};
        let decoder = decoder(ActivationDType::BF16, vec![block(attention())]);
        let head = [block(gated_selected_attention())];
        let layout = ModelStateLayout::derive(&decoder, &head.iter().collect::<Vec<_>>(), KvCodec::AffineK8V4, 0).unwrap();
        let (context, max_advance) = (64, 16);
        let plans = layout
            .head_history
            .iter()
            .map(|layout| HistoryDomainPlan { logical_rows: layout.kind().row_limit(context, max_advance).unwrap(), layout: layout.clone() })
            .collect::<Vec<_>>();
        let catalog = seismic::Devices::discover(&[]).expect("devices are discovered");
        let device = std::rc::Rc::new(catalog.open_backend(seismic::BackendName::Cpu).expect("the CPU backend opens"));
        let banks = BankCapacity { active: 4, in_flight: 4, retained: 4 };
        let build = || StateStore::new(device.clone(), context, max_advance, plans.clone(), vec![], banks.clone()).unwrap().1;
        let store = build();
        let (pending, units) = store.produced_domains(LayerRef::HeadIndexKeys { block: 0, part: 0 }).map(|(pending, units)| (pending.unwrap(), units)).unwrap();
        let rows = |ranges: Vec<(usize, usize)>| ranges.into_iter().flat_map(|(start, count)| start..start + count).collect::<Vec<_>>();
        // Two entry rows and four speculative ones: the chained row at
        // position 3 completes the first unit.
        let advance = OwnedStateAdvance::begin(store.create().unwrap(), 6).ok().unwrap();
        assert_eq!(advance.bindings().destinations[units.0].len(), 1);
        let OwnedAdvanceResolution::Committed(accepted) = advance.commit(2).ok().unwrap() else {
            panic!("a prefix commits");
        };
        let reference = build();
        let alone = OwnedStateAdvance::begin(reference.create().unwrap(), 2).ok().unwrap();
        let OwnedAdvanceResolution::Committed(alone) = alone.commit_all().ok().unwrap() else {
            panic!("an advance commits");
        };
        for domain in [pending, units] {
            assert_eq!(rows(accepted.history_ranges(domain)).len(), rows(alone.history_ranges(domain)).len());
            assert_eq!(accepted.history_start(domain), alone.history_start(domain));
            assert_eq!(store.occupied_rows(domain), reference.occupied_rows(domain));
        }
        assert_eq!(accepted.position(), 2);
        assert_eq!(store.occupied_rows(units), 0);
        // The next round completes the unit again, from the entry rows alone.
        let next = OwnedStateAdvance::begin(accepted, 2).ok().unwrap();
        assert_eq!(next.bindings().destinations[units.0].len(), 1);
    }
}
