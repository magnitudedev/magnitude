//! Device-independent row controls for state maintenance submissions.

use magnitude_model_state::{CodecConversionStep, KvCodec, PlaneCopy};
use std::{collections::HashSet, fmt};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateBatchKind {
    Copy,
    CodecConversion {
        source: KvCodec,
        destination: KvCodec,
    },
}

#[derive(Clone, Debug)]
enum StateRows {
    Copies(Vec<PlaneCopy>),
    Conversions(Vec<CodecConversionStep>),
}

/// A closed state operation: copy and conversion carry checked plane row
/// mappings.
#[derive(Clone, Debug)]
pub struct ValidatedStateBatch {
    kind: StateBatchKind,
    rows: StateRows,
    actual_rows: usize,
    class_rows: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateBatchError {
    Empty,
    InvalidMapping,
    InvalidCodec,
    Capacity { required: usize, available: usize },
}

impl fmt::Display for StateBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("state maintenance has no rows"),
            Self::InvalidMapping => formatter.write_str("state maintenance row mapping is invalid"),
            Self::InvalidCodec => formatter.write_str("state conversion requires different codecs"),
            Self::Capacity {
                required,
                available,
            } => write!(
                formatter,
                "state maintenance needs {required} rows but has {available}"
            ),
        }
    }
}

impl std::error::Error for StateBatchError {}

impl ValidatedStateBatch {
    pub fn copy(
        copies: Vec<PlaneCopy>,
        row_capacity: usize,
        class_rows: usize,
    ) -> Result<Self, StateBatchError> {
        Self::mapped(StateBatchKind::Copy, copies, row_capacity, class_rows)
    }

    pub fn convert(
        source: KvCodec,
        destination: KvCodec,
        steps: Vec<CodecConversionStep>,
        source_row_capacity: usize,
        destination_row_capacity: usize,
        class_rows: usize,
    ) -> Result<Self, StateBatchError> {
        if source == destination {
            return Err(StateBatchError::InvalidCodec);
        }
        let actual_rows = steps.first().map_or(0, |step| step.from.len());
        if actual_rows == 0 || steps.is_empty() {
            return Err(StateBatchError::Empty);
        }
        if actual_rows > class_rows {
            return Err(StateBatchError::Capacity {
                required: actual_rows,
                available: class_rows,
            });
        }
        let mut vectors = HashSet::new();
        let mut source_planes = HashSet::new();
        let mut destination_planes = HashSet::new();
        if steps.iter().any(|step| {
            step.from.len() != actual_rows
                || step.to.len() != actual_rows
                || step.source_planes.is_empty()
                || step.destination_planes.is_empty()
                || !vectors.insert((step.layer, step.vector))
                || step.from.iter().any(|row| *row >= source_row_capacity)
                || step.to.iter().any(|row| *row >= destination_row_capacity)
                || step.from.iter().copied().collect::<HashSet<_>>().len() != actual_rows
                || step.to.iter().copied().collect::<HashSet<_>>().len() != actual_rows
                || step
                    .source_planes
                    .iter()
                    .any(|plane| !source_planes.insert(*plane))
                || step
                    .destination_planes
                    .iter()
                    .any(|plane| !destination_planes.insert(*plane))
        }) {
            return Err(StateBatchError::InvalidMapping);
        }
        let first_from = &steps[0].from;
        let first_to = &steps[0].to;
        if steps
            .iter()
            .any(|step| &step.from != first_from || &step.to != first_to)
        {
            return Err(StateBatchError::InvalidMapping);
        }
        Ok(Self {
            kind: StateBatchKind::CodecConversion {
                source,
                destination,
            },
            rows: StateRows::Conversions(steps),
            actual_rows,
            class_rows,
        })
    }

    fn mapped(
        kind: StateBatchKind,
        copies: Vec<PlaneCopy>,
        row_capacity: usize,
        class_rows: usize,
    ) -> Result<Self, StateBatchError> {
        let actual_rows = copies.first().map_or(0, |copy| copy.from.len());
        if actual_rows == 0 || copies.is_empty() {
            return Err(StateBatchError::Empty);
        }
        if actual_rows > class_rows {
            return Err(StateBatchError::Capacity {
                required: actual_rows,
                available: class_rows,
            });
        }
        let mut planes = HashSet::new();
        if copies.iter().any(|copy| {
            copy.from.len() != actual_rows
                || copy.to.len() != actual_rows
                || !planes.insert(copy.plane_index)
                || copy
                    .from
                    .iter()
                    .chain(&copy.to)
                    .any(|row| *row >= row_capacity)
                || copy.from.iter().copied().collect::<HashSet<_>>().len() != actual_rows
                || copy.to.iter().copied().collect::<HashSet<_>>().len() != actual_rows
        }) {
            return Err(StateBatchError::InvalidMapping);
        }
        let first_from = &copies[0].from;
        let first_to = &copies[0].to;
        if copies
            .iter()
            .any(|copy| &copy.from != first_from || &copy.to != first_to)
        {
            return Err(StateBatchError::InvalidMapping);
        }
        Ok(Self {
            kind,
            rows: StateRows::Copies(copies),
            actual_rows,
            class_rows,
        })
    }

    pub fn kind(&self) -> StateBatchKind {
        self.kind
    }
    pub fn actual_rows(&self) -> usize {
        self.actual_rows
    }
    pub fn class_rows(&self) -> usize {
        self.class_rows
    }
    pub fn copies(&self) -> Option<&[PlaneCopy]> {
        match &self.rows {
            StateRows::Copies(copies) => Some(copies),
            StateRows::Conversions(_) => None,
        }
    }
    pub fn conversions(&self) -> Option<&[CodecConversionStep]> {
        match &self.rows {
            StateRows::Conversions(steps) => Some(steps),
            StateRows::Copies(_) => None,
        }
    }
}
