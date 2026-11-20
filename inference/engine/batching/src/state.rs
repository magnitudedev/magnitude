//! Device-independent row controls for state maintenance submissions.

use magnitude_state::{CodecConversionStep, KvCodec, PlaneCopy};
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
        source_row_capacities: &[usize],
        destination_row_capacities: &[usize],
        class_rows: usize,
    ) -> Result<Self, StateBatchError> {
        if source == destination {
            return Err(StateBatchError::InvalidCodec);
        }
        if steps.is_empty() || steps.iter().all(|step| step.from.is_empty()) {
            return Err(StateBatchError::Empty);
        }
        if source_row_capacities.len() != destination_row_capacities.len() {
            return Err(StateBatchError::InvalidMapping);
        }
        let mut vectors = HashSet::with_capacity(steps.len());
        let mut source_planes = HashSet::new();
        let mut destination_planes = HashSet::new();
        let mut domain_maps: Vec<Option<(&[usize], &[usize])>> =
            vec![None; source_row_capacities.len()];
        let mut actual_rows = 0usize;
        for step in &steps {
            let domain = step.domain.0;
            let Some(&source_capacity) = source_row_capacities.get(domain) else {
                return Err(StateBatchError::InvalidMapping);
            };
            let destination_capacity = destination_row_capacities[domain];
            if step.source_planes.is_empty()
                || step.destination_planes.is_empty()
                || !vectors.insert((step.domain, step.layer, step.vector))
                || step
                    .source_planes
                    .iter()
                    .any(|plane| !source_planes.insert(*plane))
                || step
                    .destination_planes
                    .iter()
                    .any(|plane| !destination_planes.insert(*plane))
            {
                return Err(StateBatchError::InvalidMapping);
            }
            match domain_maps[domain] {
                Some((first_from, first_to)) => {
                    if step.from.as_slice() != first_from || step.to.as_slice() != first_to {
                        return Err(StateBatchError::InvalidMapping);
                    }
                }
                None => {
                    if step.from.is_empty()
                        || step.from.len() != step.to.len()
                        || step.from.iter().any(|&row| row >= source_capacity)
                        || step.to.iter().any(|&row| row >= destination_capacity)
                        || !unique_rows(&step.from)
                        || !unique_rows(&step.to)
                    {
                        return Err(StateBatchError::InvalidMapping);
                    }
                    domain_maps[domain] = Some((step.from.as_slice(), step.to.as_slice()));
                    actual_rows = actual_rows
                        .checked_add(step.from.len())
                        .ok_or(StateBatchError::InvalidMapping)?;
                }
            }
        }
        if actual_rows > class_rows {
            return Err(StateBatchError::Capacity {
                required: actual_rows,
                available: class_rows,
            });
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
        let first_from = &copies[0].from;
        let first_to = &copies[0].to;
        if first_to.len() != actual_rows
            || first_from
                .iter()
                .chain(first_to)
                .any(|&row| row >= row_capacity)
            || !unique_rows(first_from)
            || !unique_rows(first_to)
        {
            return Err(StateBatchError::InvalidMapping);
        }
        let mut planes = HashSet::with_capacity(copies.len());
        if copies.iter().any(|copy| {
            !planes.insert(copy.plane_index)
                || &copy.from != first_from
                || &copy.to != first_to
        }) {
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

fn unique_rows(rows: &[usize]) -> bool {
    if rows.windows(2).all(|pair| pair[0] < pair[1]) {
        return true;
    }
    let mut seen = HashSet::with_capacity(rows.len());
    rows.iter().all(|&row| seen.insert(row))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_rejects_duplicate_rows_and_accepts_out_of_order_unique_rows() {
        let ok = ValidatedStateBatch::copy(
            vec![
                PlaneCopy {
                    plane_index: 0,
                    from: vec![5, 2, 7],
                    to: vec![3, 1, 0],
                },
                PlaneCopy {
                    plane_index: 1,
                    from: vec![5, 2, 7],
                    to: vec![3, 1, 0],
                },
            ],
            8,
            4,
        )
        .unwrap();
        assert_eq!(ok.actual_rows(), 3);

        let dup = ValidatedStateBatch::copy(
            vec![PlaneCopy {
                plane_index: 0,
                from: vec![5, 2, 5],
                to: vec![3, 1, 0],
            }],
            8,
            4,
        );
        assert_eq!(dup.unwrap_err(), StateBatchError::InvalidMapping);
    }
}
