use crate::batching::Demand;
use crate::{ConditioningRef, FeatureRef, ImageRef, LogitsRef};
pub use magnitude_artifacts::TokenId;
use std::{fmt, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramIdentity(Arc<str>);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceDomainId(Arc<str>);

macro_rules! identity {
    ($name:ident) => {
        impl $name {
            pub fn new(value: impl Into<Arc<str>>) -> Result<Self, String> {
                let value = value.into();
                if value.is_empty() {
                    return Err(concat!(stringify!($name), " must not be empty").into());
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

identity!(ProgramIdentity);
identity!(ResourceDomainId);

/// Engine-wide identity of an admitted request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

/// A contiguous row range within an aggregate feature lease.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureSpan {
    pub features: FeatureRef,
    pub start: usize,
    pub count: usize,
}

#[derive(Clone)]
pub struct RetainedFeatureSpan {
    span: FeatureSpan,
    bytes: u64,
}

impl RetainedFeatureSpan {
    pub fn new(span: FeatureSpan, bytes: u64) -> Result<Self, OperationError> {
        if span.count == 0 || span.start.checked_add(span.count).is_none() {
            return Err(OperationError::FeatureSpan {
                count: span.count,
                rows: span.count,
            });
        }
        if bytes == 0 {
            return Err(OperationError::FeatureStorageBytes);
        }
        Ok(Self { span, bytes })
    }

    pub fn span(&self) -> FeatureSpan {
        self.span.clone()
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl fmt::Debug for RetainedFeatureSpan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetainedFeatureSpan")
            .field("span", &self.span)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl PartialEq for RetainedFeatureSpan {
    fn eq(&self, other: &Self) -> bool {
        self.span == other.span && self.bytes == other.bytes
    }
}

impl Eq for RetainedFeatureSpan {}

/// Resource-domain seam used by generation to turn transient executor output
/// rows into independently owned device storage.
pub trait FeatureRetainer {
    fn retain(&mut self, span: FeatureSpan) -> Result<RetainedFeatureSpan, String>;
}

impl FeatureSpan {
    pub fn new(features: FeatureRef, start: usize, count: usize) -> Result<Self, OperationError> {
        if count == 0 || start.checked_add(count).is_none() {
            return Err(OperationError::FeatureSpan { count, rows: count });
        }
        Ok(Self {
            features,
            start,
            count,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkKind {
    Prefill,
    Replay,
    Decode,
    Verify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sampling {
    Greedy,
    Categorical,
}

/// Device-side distribution shaping. Identity values leave logits unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shaping {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Default for Shaping {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl Shaping {
    pub fn validate(self) -> Result<Self, &'static str> {
        let finite = [
            self.temperature,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
        ]
        .into_iter()
        .all(f32::is_finite);
        if !(finite
            && self.temperature >= 0.0
            && 0.0 < self.top_p
            && self.top_p <= 1.0
            && (0.0..=1.0).contains(&self.min_p)
            && self.top_k <= (1 << 24)
            && self.repetition_penalty > 0.0)
        {
            return Err("invalid sampling shaping parameters");
        }
        Ok(self)
    }

    pub fn is_identity(self) -> bool {
        self == Self::default()
    }

    pub fn uses_history(self) -> bool {
        self.repetition_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectSpec {
    pub sampling: Sampling,
    pub seed: u64,
    pub position: usize,
    pub domain: u32,
    pub mask: Option<Arc<[u32]>>,
    pub shaping: Shaping,
    pub history: Option<Arc<[i32]>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    Forward {
        request: RequestId,
        kind: WorkKind,
        tokens: Vec<TokenId>,
        position: usize,
        conditioning: Option<ConditioningRef>,
        demand: Demand,
        /// Selection rows in physical token order. Verification supplies one
        /// spec per token; decode supplies one for its sole row; a finishing
        /// prefill supplies one for its final row. Other rows have no spec.
        select: Vec<SelectSpec>,
        committed: usize,
    },
    Head {
        request: RequestId,
        tokens: Vec<TokenId>,
        conditioning: FeatureSpan,
        position: usize,
        demand: Demand,
    },
    Project {
        request: RequestId,
        features: FeatureRef,
        select: SelectSpec,
    },
    Repair {
        request: RequestId,
        rows: usize,
    },
    Encode {
        request: RequestId,
        image: ImageRef,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecutableKind {
    Target,
    Head,
    Encoder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CommittedClass {
    AllCommitted,
    HasTentative,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupKey {
    pub program_identity: ProgramIdentity,
    pub executable: ExecutableKind,
    pub commitment: CommittedClass,
}

impl Operation {
    pub const fn request(&self) -> RequestId {
        match self {
            Self::Forward { request, .. }
            | Self::Head { request, .. }
            | Self::Project { request, .. }
            | Self::Repair { request, .. }
            | Self::Encode { request, .. } => *request,
        }
    }

    pub const fn executable(&self) -> ExecutableKind {
        match self {
            Self::Forward { .. } | Self::Repair { .. } => ExecutableKind::Target,
            Self::Head { .. } | Self::Project { .. } => ExecutableKind::Head,
            Self::Encode { .. } => ExecutableKind::Encoder,
        }
    }

    pub fn demand(&self) -> Demand {
        match self {
            Self::Forward { demand, .. } | Self::Head { demand, .. } => *demand,
            Self::Project { .. } => Demand::SELECT,
            Self::Repair { .. } | Self::Encode { .. } => Demand::NONE,
        }
    }

    pub fn row_count(&self) -> usize {
        match self {
            Self::Forward { tokens, .. } | Self::Head { tokens, .. } => tokens.len(),
            Self::Project { .. } => 1,
            Self::Repair { rows, .. } => *rows,
            Self::Encode { image, .. } => image.patches(),
        }
    }

    pub fn commitment(&self) -> CommittedClass {
        match self {
            Self::Forward {
                tokens, committed, ..
            } if *committed < tokens.len() => CommittedClass::HasTentative,
            Self::Head { .. } | Self::Project { .. } => CommittedClass::HasTentative,
            _ => CommittedClass::AllCommitted,
        }
    }

    /// Maps a forward's compact selection table back to physical token rows.
    /// Verification maps one-to-one, decode maps its only row, and finishing
    /// prefill maps its sole selection spec to the final token row.
    pub fn selection_for_row(&self, row: usize) -> Option<&SelectSpec> {
        let Self::Forward {
            kind,
            tokens,
            select,
            ..
        } = self
        else {
            return None;
        };
        match kind {
            WorkKind::Verify => select.get(row),
            WorkKind::Decode if row == 0 => select.first(),
            WorkKind::Prefill if row.checked_add(1) == Some(tokens.len()) => select.first(),
            WorkKind::Decode | WorkKind::Prefill | WorkKind::Replay => None,
        }
    }

    pub fn validate(&self) -> Result<(), OperationError> {
        if self.row_count() == 0 {
            return Err(OperationError::EmptyRows);
        }
        match self {
            Self::Forward {
                kind,
                tokens,
                demand,
                select,
                committed,
                ..
            } => {
                if *committed > tokens.len() {
                    return Err(OperationError::CommittedRows {
                        committed: *committed,
                        rows: tokens.len(),
                    });
                }
                validate_forward_select(*kind, tokens.len(), *demand, select)?;
            }
            Self::Project { select, .. } => validate_select(select)?,
            Self::Head {
                tokens,
                conditioning,
                ..
            } => {
                if conditioning.count != tokens.len()
                    || conditioning.count == 0
                    || conditioning.start.checked_add(conditioning.count).is_none()
                {
                    return Err(OperationError::FeatureSpan {
                        count: conditioning.count,
                        rows: tokens.len(),
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn validate_forward_select(
    kind: WorkKind,
    rows: usize,
    demand: Demand,
    select: &[SelectSpec],
) -> Result<(), OperationError> {
    if demand.contains(Demand::SELECT) == select.is_empty() {
        return Err(OperationError::SelectionMismatch);
    }
    let expected = match kind {
        WorkKind::Verify => rows,
        WorkKind::Decode if demand.contains(Demand::SELECT) && rows == 1 => 1,
        WorkKind::Decode if demand.contains(Demand::SELECT) => {
            return Err(OperationError::SelectingDecodeWidth(rows));
        }
        WorkKind::Prefill if demand.contains(Demand::SELECT) => 1,
        WorkKind::Decode | WorkKind::Prefill | WorkKind::Replay => 0,
    };
    if select.len() != expected {
        return Err(OperationError::SelectionRows {
            kind,
            expected,
            actual: select.len(),
        });
    }
    for spec in select {
        validate_select(spec)?;
    }
    Ok(())
}

fn validate_select(select: &SelectSpec) -> Result<(), OperationError> {
    select
        .shaping
        .validate()
        .map_err(|_| OperationError::InvalidShaping)?;
    if select.position > i32::MAX as usize {
        return Err(OperationError::SelectionPosition(select.position));
    }
    if select.mask.as_ref().is_some_and(|mask| mask.is_empty()) {
        return Err(OperationError::EmptyMask);
    }
    match select.history.as_deref() {
        Some(history) if history.len() != 64 => {
            return Err(OperationError::HistoryWidth(history.len()));
        }
        None if select.shaping.uses_history() => {
            return Err(OperationError::MissingHistory);
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationError {
    EmptyRows,
    CommittedRows {
        committed: usize,
        rows: usize,
    },
    SelectionMismatch,
    SelectingDecodeWidth(usize),
    SelectionRows {
        kind: WorkKind,
        expected: usize,
        actual: usize,
    },
    InvalidShaping,
    SelectionPosition(usize),
    EmptyMask,
    MissingHistory,
    HistoryWidth(usize),
    FeatureSpan {
        count: usize,
        rows: usize,
    },
    FeatureStorageBytes,
}

impl fmt::Display for OperationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRows => formatter.write_str("operation requires at least one row"),
            Self::CommittedRows { committed, rows } => write!(
                formatter,
                "operation declares {committed} committed rows for a width of {rows}"
            ),
            Self::SelectionMismatch => formatter.write_str(
                "selection demand and selection parameters must either both be present or absent",
            ),
            Self::SelectingDecodeWidth(width) => write!(
                formatter,
                "a selecting Decode must contain one row, not {width}",
            ),
            Self::SelectionRows {
                kind,
                expected,
                actual,
            } => write!(
                formatter,
                "{kind:?} forward has {actual} selection rows; expected {expected}"
            ),
            Self::InvalidShaping => formatter.write_str("selection shaping parameters are invalid"),
            Self::SelectionPosition(position) => write!(
                formatter,
                "selection position {position} exceeds the packed i32 domain"
            ),
            Self::EmptyMask => formatter.write_str("selection mask cannot be empty"),
            Self::MissingHistory => {
                formatter.write_str("history-dependent shaping requires a 64-token history row")
            }
            Self::HistoryWidth(width) => write!(
                formatter,
                "selection history has width {width}; expected 64",
            ),
            Self::FeatureSpan { count, rows } => write!(
                formatter,
                "head conditioning span has {count} rows for {rows} token rows",
            ),
            Self::FeatureStorageBytes => {
                formatter.write_str("retained feature storage must charge at least one byte")
            }
        }
    }
}

impl std::error::Error for OperationError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selected {
    pub token: TokenId,
    pub status: u8,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct RowResult {
    pub selected: Option<Selected>,
    pub features: Option<FeatureRef>,
    pub logits: Option<LogitsRef>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Forward { rows: Vec<RowResult> },
    Head { features: FeatureRef },
    Project { selected: Vec<Selected> },
    Repair,
    Encode { features: FeatureRef },
}
