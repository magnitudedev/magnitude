//! History domains. A history domain is a set of attention layers whose
//! history shares one row numbering: a row is one token's history in the
//! domain's layers. A store holds one history slab tensor per stored domain,
//! with its own components, placement geometry (pages, slabs and span limit),
//! free space and per-row references.

use crate::{ComponentDescriptor, Error, HistoryGeometry, LayerRef, LayoutError, history_geometry};
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
            Self::Block { rate } => (position + count) / rate - position / rate,
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

    /// The placement geometry of this stored domain among a store's
    /// `domains` (see [`history_geometry`]): its histories hold at most its
    /// row limit, and Shared layers reading it also see one advance's
    /// appended rows.
    pub fn geometry(
        &self,
        domains: &[HistoryDomainLayout],
        context: usize,
        max_advance: usize,
    ) -> Result<HistoryGeometry, Error> {
        history_geometry(
            self.row_bytes()?,
            self.kind().row_limit(context, max_advance)?,
            prepared_appended_rows(self.kind(), self.components(), domains, max_advance),
        )
        .map_err(Error::Request)
    }
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
    pub geometry: HistoryGeometry,
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
    for plan in plans {
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
                        .any(|component| !matches!(component.layer, LayerRef::Pooled { .. })),
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
                let mut row_bytes = 0u64;
                for component in &components {
                    if !layers.insert((component.layer, matches!(kind, HistoryDomainKind::Block { .. }))) {
                        return Err(LayoutError::DuplicateLayer(component.layer).into());
                    }
                    for plane in component.planes() {
                        let bytes = u64::try_from(plane.row_bytes)
                            .map_err(|_| LayoutError::ArithmeticOverflow("plane row bytes"))?;
                        row_bytes = row_bytes
                            .checked_add(bytes)
                            .ok_or(LayoutError::ArithmeticOverflow("history row bytes"))?;
                    }
                }
                let geometry = history_geometry(
                    row_bytes,
                    row_limit,
                    prepared_appended_rows(kind, &components, &layouts, max_advance),
                )
                .map_err(Error::Request)?;
                // Every plane's capacity is at most the domain's total.
                let logical_rows = geometry
                    .reserved_rows(plan.logical_rows)
                    .filter(|&rows| row_bytes.checked_mul(rows as u64).is_some())
                    .ok_or(LayoutError::ArithmeticOverflow("total history capacity"))?;
                stored.push(StoredDomain {
                    kind,
                    components,
                    row_bytes,
                    geometry,
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
