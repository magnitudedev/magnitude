//! History domains. A history domain is a set of attention layers whose
//! history shares one row numbering: a row is one token's history in the
//! domain's layers. A store holds one history slab tensor per stored domain,
//! with its own components, rows per slab, span bound, free space and per-row
//! references.

use crate::{
    history_rows_per_slab, max_visible_spans, ComponentDescriptor, Error, LayerRef, LayoutError,
};
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
    /// One row per `rate` tokens, committed at block boundaries. Interface
    /// only: a store rejects it.
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
            Self::Block { .. } => Err(Error::UnsupportedHistoryDomain(self)),
        }
    }

    /// Rows a checkpoint or retained prefix at position `position`
    /// references: every row for Token, rows `[position - n, position)` for
    /// Window(n).
    pub fn checkpoint_rows(self, position: usize) -> Result<usize, Error> {
        match self {
            Self::Token => Ok(position),
            Self::Window { rows } => Ok(rows.min(position)),
            Self::Shared { .. } => Ok(0),
            Self::Block { .. } => Err(Error::UnsupportedHistoryDomain(self)),
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
    /// 0 for Token, `position - n` for Window(n).
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

    /// The slab-rounded steady footprint of `live` histories and
    /// `checkpoints` checkpoints at position `context`, for assessment. A
    /// live history holds its row limit (for Window(n), `n` plus one advance
    /// whatever the context); a checkpoint holds its checkpoint rows. Rows a
    /// checkpoint shares with a live history are counted for both, so the
    /// footprint is an upper bound. Rows of distinct histories share slabs,
    /// so rounding applies to the domain total. Shared domains hold nothing.
    pub fn steady_footprint(
        &self,
        context: usize,
        max_advance: usize,
        live: u64,
        checkpoints: u64,
    ) -> Result<HistoryFootprint, Error> {
        let kind = self.kind();
        if let HistoryDomainKind::Shared { .. } = kind {
            return Ok(HistoryFootprint::default());
        }
        let overflow = || Error::Layout(LayoutError::ArithmeticOverflow("history footprint"));
        let live_rows =
            u64::try_from(kind.row_limit(context, max_advance)?).map_err(|_| overflow())?;
        let checkpoint_rows =
            u64::try_from(kind.checkpoint_rows(context)?).map_err(|_| overflow())?;
        let rows = live
            .checked_mul(live_rows)
            .and_then(|live| checkpoints.checked_mul(checkpoint_rows)?.checked_add(live))
            .ok_or_else(overflow)?;
        let rows_per_slab = history_rows_per_slab(self.row_bytes()?).map_err(Error::Request)?;
        Ok(HistoryFootprint {
            rows_per_slab,
            rows,
            slabs: rows.div_ceil(rows_per_slab as u64),
        })
    }
}

/// A domain's steady history in rows and whole slabs. The planner prices
/// the slabs with the Seismic slab layout of the domain's regions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HistoryFootprint {
    pub rows_per_slab: usize,
    pub rows: u64,
    pub slabs: u64,
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
    pub slab_rows: usize,
    pub logical_rows: usize,
    pub span_bound: usize,
}

/// Where a layer's history lives: the stored domain and the layer whose
/// regions it binds (itself, or a Shared domain's source).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistorySource {
    pub domain: HistoryDomainId,
    pub layer: LayerRef,
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
    let mut layers = BTreeSet::new();
    let mut shared = Vec::new();
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
            HistoryDomainLayout::Block { .. } => {
                return Err(Error::UnsupportedHistoryDomain(kind));
            }
            HistoryDomainLayout::Token { components }
            | HistoryDomainLayout::Window { components, .. } => {
                if components.is_empty() {
                    return Err(LayoutError::EmptyHistoryDomain.into());
                }
                if kind == (HistoryDomainKind::Window { rows: 0 }) {
                    return Err(LayoutError::ZeroWindow.into());
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
                    if !layers.insert(component.layer) {
                        return Err(LayoutError::DuplicateLayer(component.layer).into());
                    }
                    for plane in component.planes() {
                        let bytes = u64::try_from(plane.row_bytes)
                            .map_err(|_| LayoutError::ArithmeticOverflow("plane row bytes"))?;
                        bytes
                            .checked_mul(plan.logical_rows as u64)
                            .ok_or(LayoutError::ArithmeticOverflow("history plane capacity"))?;
                        row_bytes = row_bytes
                            .checked_add(bytes)
                            .ok_or(LayoutError::ArithmeticOverflow("history row bytes"))?;
                    }
                }
                row_bytes
                    .checked_mul(plan.logical_rows as u64)
                    .ok_or(LayoutError::ArithmeticOverflow("total history capacity"))?;
                let slab_rows = history_rows_per_slab(row_bytes).map_err(Error::Request)?;
                stored.push(StoredDomain {
                    kind,
                    components,
                    row_bytes,
                    slab_rows,
                    logical_rows: plan.logical_rows,
                    span_bound: max_visible_spans(row_limit, slab_rows).map_err(Error::Request)?,
                });
            }
        }
    }
    let mut bindings = Vec::new();
    for (source, readers) in shared {
        let domain = stored
            .iter()
            .position(|domain| {
                domain
                    .components
                    .iter()
                    .any(|component| component.layer == source)
            })
            .ok_or(LayoutError::UnknownSharedSource(source))?;
        for layer in readers {
            if !layers.insert(layer) {
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
