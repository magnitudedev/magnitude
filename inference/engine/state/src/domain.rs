//! History domains. A history domain is a set of attention layers whose
//! history shares one row numbering: a row is one token's history in the
//! domain's layers. A store holds one history slab tensor per stored domain,
//! with its own components, placement geometry (pages, slabs and span limit),
//! free space and per-row references.

use crate::{ComponentDescriptor, Error, HistoryGeometry, HistoryPages, LayerRef, LayoutError, history_pages};
use seismic::{Element, SlabRegion};
use std::collections::BTreeSet;

/// A stored history domain of one store: its index among the store's stored
/// domains, in plan order. Shared domains hold no storage and have no index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HistoryDomainId(pub usize);

/// Which rows a history of the domain references.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HistoryDomainKind {
    /// One row per token for the whole context.
    Token,
    /// One row per token, but a history references only its last `rows`
    /// accepted rows plus its tentative rows.
    Window { rows: usize },
    /// One row per `rate` tokens: block `b`'s row is written by the advance
    /// whose rows reach position `(b + 1) rate`, and a history at position
    /// `p` references blocks `[0, p / rate)`.
    Block { rate: usize },
    /// No storage: the domain's layers read the regions of layer `source`.
    Shared { source: LayerRef },
}

impl HistoryDomainKind {
    /// The most rows one history references at once: its accepted rows at
    /// `context` tokens plus one advance of at most `max_advance` tentative
    /// rows. A domain's span bound follows from it.
    pub fn row_limit(self, context: usize, max_advance: usize) -> Result<usize, Error> {
        match self {
            Self::Token => Ok(context),
            Self::Window { rows } => Ok(rows
                .checked_add(max_advance)
                .ok_or(LayoutError::ArithmeticOverflow("window row limit"))?
                .min(context)),
            Self::Shared { .. } => Ok(0),
            Self::Block { rate } => Ok(context / rate),
        }
    }

    /// Rows a checkpoint or retained prefix at position `position`
    /// references: every row for Token, rows `[position - n, position)` for
    /// Window(n), every complete block for Block.
    pub fn checkpoint_rows(self, position: usize) -> Result<usize, Error> {
        match self {
            Self::Token => Ok(position),
            Self::Window { rows } => Ok(rows.min(position)),
            Self::Shared { .. } => Ok(0),
            Self::Block { rate } => Ok(position / rate),
        }
    }

    /// The domain's rows an advance of `count` rows from `position` adds:
    /// one per row, or for Block one per block it completes.
    pub fn advance_rows(self, position: usize, count: usize) -> usize {
        match self {
            Self::Block { rate } => {
                let end = position as u128 + count as u128;
                (end / rate as u128 - position as u128 / rate as u128) as usize
            }
            Self::Token | Self::Window { .. } | Self::Shared { .. } => count,
        }
    }

    /// The most rows an advance of `count` rows adds to the domain from any
    /// position ([`Self::advance_rows`] at its worst position): one per row,
    /// or for Block every block the rows can complete, `ceil(count / rate)`.
    pub fn most_advance_rows(self, count: usize) -> usize {
        match self {
            Self::Block { rate } => count.div_ceil(rate),
            Self::Token | Self::Window { .. } | Self::Shared { .. } => count,
        }
    }

    /// The first position a query at `position` reads: 0 for Token, and for
    /// Window(n) the start of the `n` most recent tokens including the query
    /// itself (a key at `k` is read while `position - k < n`).
    pub fn visible_from(self, position: usize) -> usize {
        match self {
            Self::Window { rows } => (position + 1).saturating_sub(rows),
            Self::Token | Self::Block { .. } | Self::Shared { .. } => 0,
        }
    }

    /// The first logical position a history at `position` still references:
    /// 0 for Token and Block, `position - n` for Window(n).
    pub(crate) fn retained_from(self, position: usize) -> usize {
        match self {
            Self::Window { rows } => position.saturating_sub(rows),
            Self::Token | Self::Block { .. } | Self::Shared { .. } => 0,
        }
    }
}

/// The device-free layout of one history domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HistoryDomainLayout {
    Token {
        components: Vec<ComponentDescriptor>,
    },
    Window {
        rows: usize,
        components: Vec<ComponentDescriptor>,
    },
    Block {
        rate: usize,
        components: Vec<ComponentDescriptor>,
    },
    /// `layers` bind the slab regions of layer `source`, a layer of a stored
    /// domain, and see that domain's visible spans.
    Shared {
        source: LayerRef,
        layers: Vec<LayerRef>,
    },
}

impl HistoryDomainLayout {
    pub fn kind(&self) -> HistoryDomainKind {
        match self {
            Self::Token { .. } => HistoryDomainKind::Token,
            Self::Window { rows, .. } => HistoryDomainKind::Window { rows: *rows },
            Self::Block { rate, .. } => HistoryDomainKind::Block { rate: *rate },
            Self::Shared { source, .. } => HistoryDomainKind::Shared { source: *source },
        }
    }

    /// The components stored in the domain's slabs; none for Shared.
    pub fn components(&self) -> &[ComponentDescriptor] {
        match self {
            Self::Token { components }
            | Self::Window { components, .. }
            | Self::Block { components, .. } => components,
            Self::Shared { .. } => &[],
        }
    }

    /// Bytes of one row: every component part of every layer in the domain.
    pub fn row_bytes(&self) -> Result<u64, LayoutError> {
        self.components().iter().try_fold(0u64, |total, component| {
            let bytes = u64::try_from(component.row_bytes()?)
                .map_err(|_| LayoutError::ArithmeticOverflow("history row bytes"))?;
            total
                .checked_add(bytes)
                .ok_or(LayoutError::ArithmeticOverflow("history row bytes"))
        })
    }

    /// The pages of this stored domain among a store's `domains` (see
    /// [`history_pages`]): its histories hold at most its row limit, and
    /// Shared layers reading it also see one advance's appended rows.
    pub fn pages(
        &self,
        domains: &[HistoryDomainLayout],
        context: usize,
        max_advance: usize,
    ) -> Result<HistoryPages, Error> {
        Ok(history_pages(
            self.kind().row_limit(context, max_advance)?,
            prepared_appended_rows(self.kind(), self.components(), domains, max_advance),
        ))
    }

    /// The placement geometry of this stored domain among a store's
    /// `domains` in a store of `granule`-byte granules: its pages
    /// ([`Self::pages`]) in slabs of as many as fit a granule
    /// ([`HistoryPages::in_granule`]).
    pub fn geometry(
        &self,
        domains: &[HistoryDomainLayout],
        context: usize,
        max_advance: usize,
        granule: u64,
    ) -> Result<HistoryGeometry, Error> {
        self.pages(domains, context, max_advance)?
            .in_granule(self.row_bytes()?, granule)
            .map_err(Error::Request)
    }
}

/// The slab regions of a stored domain's `components`: one per plane, the
/// layout the store's slabs of the domain take.
pub fn slab_regions(components: &[ComponentDescriptor]) -> Vec<SlabRegion> {
    components
        .iter()
        .flat_map(ComponentDescriptor::planes)
        .map(|plane| SlabRegion {
            element: Element::dense(plane.dtype),
            row_shape: plane.row_extents.iter().map(|&size| size as u64).collect(),
        })
        .collect()
}

/// The state `sequences` histories of `depth` rows each take in a store of
/// `domains` at `context` tokens and advances of at most `max_advance` rows,
/// backed in `granule`-byte granules, a slab each: the page rule, the one
/// place state bytes come from. The load, the assessment, the planner and
/// the runtime ledger all size state here, a request's reservation
/// included, so what is reserved is what the store backs.
///
/// One history takes, in each stored domain, the whole pages its rows take
/// (`page_demand_rows` at the domain's pages, [`HistoryDomainLayout::pages`])
/// at its row's bytes, summed. Counted
/// from a page boundary, this is also the most a submission batch of
/// `depth` rows newly takes: a batch straddling a page boundary at `offset`
/// rows into its last page fills `page_rows - offset` rows in place first,
/// so its new pages, `ceil((depth - (page_rows - offset)) / page_rows)`,
/// are never more than at offset 0's `ceil(depth / page_rows)`. A Block
/// domain's rows are at their most blocks
/// ([`HistoryDomainKind::most_advance_rows`]) at a boundary too. A Window
/// domain's pages are capped at what its history ever occupies
/// ([`crate::layout::window_pages`] of its row limit), since it releases the
/// rows before its window as it slides. Each stored domain grows in whole
/// slabs of its own geometry ([`HistoryDomainLayout::geometry`]), so the
/// domain's pages for every sequence are rounded to its slabs before the
/// domains are summed. A store backs pages, not positions: a slab is added
/// only for pages its backed slabs' free pages lack (`HistoryDomain::admission`),
/// so a window sliding across a slab boundary takes no slab its pages do
/// not. No recurrent state is counted: a sequence's bank is its own,
/// whatever its rows (`recurrent_bank_bytes`).
pub fn state_bytes(
    domains: &[HistoryDomainLayout],
    context: usize,
    max_advance: usize,
    granule: u64,
    sequences: u64,
    depth: u64,
) -> Result<u64, Error> {
    let overflow = || Error::from(LayoutError::ArithmeticOverflow("state bytes"));
    let rows = usize::try_from(depth).map_err(|_| overflow())?;
    domains
        .iter()
        .filter(|domain| !domain.components().is_empty())
        .try_fold(0u64, |total, domain| {
            let page_rows = domain.pages(domains, context, max_advance)?.page_rows;
            let kind = domain.kind();
            let taken = crate::layout::page_demand_rows(page_rows, 0, kind.most_advance_rows(rows))
                .ok_or_else(overflow)?;
            let taken = match kind {
                HistoryDomainKind::Window { .. } => {
                    let cap = crate::layout::window_pages(page_rows, kind.row_limit(context, max_advance)?)
                        .checked_mul(page_rows)
                        .ok_or_else(overflow)?;
                    taken.min(cap)
                }
                HistoryDomainKind::Token | HistoryDomainKind::Shared { .. } | HistoryDomainKind::Block { .. } => taken,
            };
            let taken = u64::try_from(taken).ok().and_then(|taken| taken.checked_mul(sequences)).ok_or_else(overflow)?;
            let slab_rows = domain.geometry(domains, context, max_advance, granule)?.slab_rows;
            let slab_rows = u64::try_from(slab_rows).map_err(|_| overflow())?;
            let bytes = taken.div_ceil(slab_rows).checked_mul(granule);
            bytes.and_then(|bytes| total.checked_add(bytes)).ok_or_else(overflow)
        })
}

/// Whether a domain of `kind` holding `components` holds produced rows:
/// every Block domain (pooled rows and their indexer keys), and a Window of
/// produced components (a pooled source's incomplete groups, an indexer's
/// raw keys). A launch's rows read such a domain's rows of their own launch
/// as stored, after they are written, rather than fresh.
pub fn produced_domain(kind: HistoryDomainKind, components: &[ComponentDescriptor]) -> bool {
    matches!(kind, HistoryDomainKind::Block { .. })
        || components.iter().any(|component| !component.attended())
}

/// Rows within an advance read through stored history by Shared layers,
/// compressed causal target attention or an indexer. Each requires room for
/// its prepared destination spans beside the already accepted spans.
fn prepared_appended_rows(
    kind: HistoryDomainKind,
    components: &[ComponentDescriptor],
    domains: &[HistoryDomainLayout],
    max_advance: usize,
) -> usize {
    // A producer reads the pending projections and an indexer scores the
    // rows and keys of its launch's earlier rows beside the accepted ones.
    if let HistoryDomainKind::Block { rate } = kind {
        return max_advance.div_ceil(rate);
    }
    if !components.iter().all(ComponentDescriptor::attended) {
        return max_advance;
    }
    let read = domains.iter().any(|domain| match domain {
        HistoryDomainLayout::Shared { source, .. } => components
            .iter()
            .any(|component| component.layer == *source),
        _ => false,
    });
    if read
        || components
            .iter()
            .any(ComponentDescriptor::causal_rows_use_cache)
    {
        max_advance
    } else {
        0
    }
}

/// One domain of a store: its layout and the row addresses reserved for
/// it (the logical rows graphs are sealed over). A Shared domain reserves
/// none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryDomainPlan {
    pub layout: HistoryDomainLayout,
    pub logical_rows: usize,
}

/// A stored domain after validation.
pub(crate) struct StoredDomain {
    pub kind: HistoryDomainKind,
    pub components: Vec<ComponentDescriptor>,
    pub row_bytes: u64,
    pub pages: HistoryPages,
    pub logical_rows: usize,
}

/// Where a layer's history lives: the stored domain and the layer whose
/// regions it binds (itself, or a Shared domain's source).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistorySource {
    pub domain: HistoryDomainId,
    pub layer: LayerRef,
}

/// A store's history reads: what a launch row holds one control set of
/// each of. Every stored domain in plan order, then every stored domain a
/// Shared layer reads, ascending: a Shared layer appends nothing, so it
/// reads its source domain's accepted rows and the rows its source appended
/// for the advance, one read beyond the stored domains per such domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryReads {
    /// The store's stored domains.
    stored: usize,
    /// The stored domains Shared layers read, ascending.
    sources: Vec<HistoryDomainId>,
    /// Each layer with attended history and its read: a stored layer's
    /// domain holding its attended rows, a Shared layer's source domain's
    /// shared read.
    layers: Vec<(LayerRef, usize)>,
}

impl HistoryReads {
    /// The reads of a store of `plans`, as [`crate::StateStore::new`] takes
    /// them with its `context` and largest advance.
    pub fn of(plans: &[HistoryDomainPlan], context: usize, max_advance: usize) -> Result<Self, Error> {
        let (stored, shared) = validate_domains(plans.to_vec(), context, max_advance)?;
        Ok(Self::new(&stored, &shared))
    }

    /// The reads of `stored` domains and `shared` layers, as
    /// [`validate_domains`] resolved them.
    pub(crate) fn new(stored: &[StoredDomain], shared: &[(LayerRef, HistorySource)]) -> Self {
        let mut layers = Vec::new();
        for (domain, plan) in stored.iter().enumerate() {
            for component in plan.components.iter().filter(|component| component.attended()) {
                if !layers.iter().any(|(layer, _)| *layer == component.layer) {
                    layers.push((component.layer, domain));
                }
            }
        }
        let mut sources = shared
            .iter()
            .map(|(_, source)| source.domain)
            .collect::<Vec<_>>();
        sources.sort_unstable();
        sources.dedup();
        layers.extend(shared.iter().map(|(layer, source)| {
            (
                *layer,
                stored.len() + sources.partition_point(|of| *of < source.domain),
            )
        }));
        Self {
            stored: stored.len(),
            sources,
            layers,
        }
    }

    /// The reads of a launch row.
    pub fn len(&self) -> usize {
        self.stored + self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The read `layer`'s attention takes (an index below [`Self::len`]).
    /// `None` for a layer without history in the store.
    pub fn read(&self, layer: LayerRef) -> Option<usize> {
        self.layers
            .iter()
            .find(|(of, _)| *of == layer)
            .map(|&(_, read)| read)
    }

    /// The stored domains Shared layers read, ascending: the reads after
    /// the stored domains'.
    pub fn shared_sources(&self) -> &[HistoryDomainId] {
        &self.sources
    }
}

/// Validate a store's domains: stored domains own nonempty, distinct layers
/// and reserve at least their row limit; every Shared layer is distinct and
/// reads a stored layer. Returns the stored domains in plan order and the
/// Shared layers with their sources.
pub(crate) fn validate_domains(
    plans: Vec<HistoryDomainPlan>,
    context: usize,
    max_advance: usize,
) -> Result<(Vec<StoredDomain>, Vec<(LayerRef, HistorySource)>), Error> {
    let mut stored = Vec::new();
    // A layer has one component per domain kind: a produced layer may hold
    // a Window's (its pending projections) and a Block's (its rows or keys).
    let mut layers = BTreeSet::new();
    let mut shared = Vec::new();
    let layouts = plans
        .iter()
        .map(|plan| plan.layout.clone())
        .collect::<Vec<_>>();
    for (plan, layout) in plans.into_iter().zip(&layouts) {
        let kind = plan.layout.kind();
        match plan.layout {
            HistoryDomainLayout::Shared {
                source,
                layers: readers,
            } => {
                if plan.logical_rows != 0 {
                    return Err(LayoutError::SharedDomainRows(source).into());
                }
                if readers.is_empty() {
                    return Err(LayoutError::EmptyHistoryDomain.into());
                }
                shared.push((source, readers));
            }
            HistoryDomainLayout::Token { components }
            | HistoryDomainLayout::Window { components, .. }
            | HistoryDomainLayout::Block { components, .. } => {
                if components.is_empty() {
                    return Err(LayoutError::EmptyHistoryDomain.into());
                }
                if kind == (HistoryDomainKind::Window { rows: 0 })
                    || kind == (HistoryDomainKind::Block { rate: 0 })
                {
                    return Err(LayoutError::ZeroWindow.into());
                }
                // Per-token attended rows (a Token or Window domain of them)
                // and produced rows keep apart: a produced domain holds a
                // pooled source's incomplete groups or an indexer's raw keys
                // (Window), or pooled rows and keys (Block), and follows
                // every per-token one, whose rows a launch reads causally.
                let produced = produced_domain(kind, &components);
                let attended = components.iter().filter(|component| component.attended());
                let mixed = match kind {
                    // A Block domain's attended rows are pooled sources'.
                    HistoryDomainKind::Block { .. } => attended
                        .clone()
                        .any(|component| !component.layer.is_pooled()),
                    _ => produced && attended.clone().next().is_some(),
                };
                if mixed
                    || (!produced
                        && stored.iter().any(|domain: &StoredDomain| {
                            produced_domain(domain.kind, &domain.components)
                        }))
                {
                    return Err(LayoutError::WrongRepresentation(
                        "produced rows in their own domains after the per-token ones",
                    )
                    .into());
                }
                let row_limit = kind.row_limit(context, max_advance)?;
                if plan.logical_rows < row_limit {
                    return Err(Error::Request(format!(
                        "history domain reserves {} rows below its row limit {row_limit}",
                        plan.logical_rows
                    )));
                }
                for component in &components {
                    if !layers.insert((component.layer, matches!(kind, HistoryDomainKind::Block { .. }))) {
                        return Err(LayoutError::DuplicateLayer(component.layer).into());
                    }
                }
                let row_bytes = layout.row_bytes()?;
                let pages = layout.pages(&layouts, context, max_advance)?;
                // Every plane's capacity is at most the domain's total.
                let logical_rows = pages
                    .reserved_rows(plan.logical_rows)
                    .filter(|&rows| row_bytes.checked_mul(rows as u64).is_some())
                    .ok_or(LayoutError::ArithmeticOverflow("total history capacity"))?;
                stored.push(StoredDomain {
                    kind,
                    components,
                    row_bytes,
                    pages,
                    logical_rows,
                });
            }
        }
    }
    let mut bindings = Vec::new();
    for (source, readers) in shared {
        // A Shared layer reads its source's attended rows (a pooled
        // source's Block rows, not its incomplete groups).
        let domain = stored
            .iter()
            .position(|domain| {
                domain
                    .components
                    .iter()
                    .any(|component| component.layer == source && component.attended())
            })
            .ok_or(LayoutError::UnknownSharedSource(source))?;
        for layer in readers {
            if !layers.insert((layer, false)) {
                return Err(LayoutError::DuplicateLayer(layer).into());
            }
            bindings.push((
                layer,
                HistorySource {
                    domain: HistoryDomainId(domain),
                    layer: source,
                },
            ));
        }
    }
    Ok((stored, bindings))
}
