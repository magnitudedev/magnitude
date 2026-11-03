//! Final normalized hidden states that sequences retain for scores readouts.
//!
//! A scores readout reads the final normalized hidden states of rows a
//! sequence computed in earlier steps, or that the sequence it was forked
//! from computed. A sequence declares the rows it will retain
//! ([`ExecutorDomain::retain_features`]) before computing them; each step
//! that computes one keeps its state in the sequence's store on the device,
//! and the step that computes the last row a scores readout reads answers it
//! from the stores. No hidden state reaches the host.

use super::*;
use magnitude_readout::{Readout, ReadoutKind};

/// Rows of one sequence whose final normalized hidden states it retains:
/// store row `i` holds sequence row `rows[i]`, as `[rows, hidden]`
/// activations on the device. Only the sequence that declared it writes it;
/// it and the sequences forked from it read it.
#[derive(Debug)]
pub(crate) struct FeatureStore {
    rows: Vec<usize>,
    features: FeatureRef,
}

impl FeatureStore {
    pub(crate) fn features(&self) -> &FeatureRef {
        &self.features
    }
}

/// Where a sequence's retained rows are: its stores, those of the sequence
/// it was forked from first and its own last.
pub(super) struct RetainedRows<'a>(pub(super) &'a [Rc<FeatureStore>]);

impl RetainedRows<'_> {
    /// The store and store row of sequence row `row`.
    fn locate(&self, row: usize) -> Option<(u32, u32)> {
        self.0.iter().enumerate().find_map(|(store, retained)| {
            let index = retained.rows.binary_search(&row).ok()?;
            Some((u32::try_from(store).ok()?, u32::try_from(index).ok()?))
        })
    }

    /// Whether row `row` is one the sequence stores itself, at which store
    /// row: its own store is the last.
    fn own(&self, row: usize) -> Option<(u32, u32)> {
        let store = self.0.len().checked_sub(1)?;
        let index = self.0[store].rows.binary_search(&row).ok()?;
        Some((u32::try_from(store).ok()?, u32::try_from(index).ok()?))
    }

    /// The batch's form of a scores readout of `rows` at `query`: the query's
    /// store row, and the keys as runs of rows of one store each. Every row
    /// must be retained.
    fn scores(
        &self,
        query: usize,
        rows: &std::ops::Range<usize>,
    ) -> Result<crate::batching::RowReadout, String> {
        let unretained = || format!("a scores readout reads row {query} or {rows:?}, unretained");
        let query = self.locate(query).ok_or_else(unretained)?;
        let mut keys = Vec::<(u32, std::ops::Range<u32>)>::new();
        for row in rows.clone() {
            let (store, index) = self.locate(row).ok_or_else(unretained)?;
            match keys.last_mut() {
                Some((last, run)) if *last == store && run.end == index => run.end += 1,
                _ => keys.push((store, index..index + 1)),
            }
        }
        Ok(crate::batching::RowReadout::Scores { query, keys })
    }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Retain the final normalized hidden states of `rows` of `request`
    /// (ascending, none computed yet) as its steps compute them, for its
    /// scores readouts and those of sequences forked from it. With `from`,
    /// the sequence `request` was forked from, `request` also reads the rows
    /// `from` retains. Declared once per sequence; the store is device memory
    /// held until the sequence closes.
    pub fn retain_features(
        &mut self,
        request: RequestId,
        rows: Vec<usize>,
        from: Option<RequestId>,
    ) -> Result<(), DomainError> {
        let position = self
            .target
            .get(&request)
            .ok_or_else(|| DomainError::invariant("a retaining request is not resident"))?
            .position();
        if self.retained.contains_key(&request)
            || rows.first().is_none_or(|first| *first < position)
            || !rows.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err(DomainError::Input(
                "retained rows are ascending rows of a resident sequence it has not computed, \
                 declared once"
                    .into(),
            ));
        }
        let mut stores = match from {
            Some(from) => self
                .retained
                .get(&from)
                .cloned()
                .ok_or_else(|| DomainError::invariant("a forked sequence's source retains no rows"))?,
            None => Vec::new(),
        };
        let decoder = &self.definition.decoder;
        let element =
            seismic::Element::dense(crate::planning::activation_dtype(decoder.activation_dtype));
        let extents = [rows.len() as u64, decoder.hidden];
        let bytes = element
            .canonical_byte_len(&extents)
            .map_err(|error| DomainError::invariant(error.to_string()))?;
        let claim = self.claim_device_growth(bytes, 0, HoldingClass::Live)?;
        let tensor = Tensor::zeros(self.domain.device(), element, &extents);
        drop(claim);
        let tensor = tensor.map_err(|error| DomainError::Input(error.to_string()))?;
        let features = self
            .domain
            .publish_features(tensor)
            .map_err(|error| DomainError::Input(error.to_string()))?;
        stores.push(Rc::new(FeatureStore { rows, features }));
        self.retained.insert(request, stores);
        Ok(())
    }

    /// `operation`'s demand of its launch: a forward whose chunk computes a
    /// row its sequence retains also reads out (`RowReadout::Retain`).
    pub(super) fn launch_demand(&self, operation: &Operation) -> crate::batching::Demand {
        let retains = match operation {
            Operation::Forward {
                request,
                position,
                tokens,
                ..
            } => self.retained_rows(*request).0.last().is_some_and(|own| {
                own.rows
                    .iter()
                    .any(|row| (*position..position + tokens.len()).contains(row))
            }),
            Operation::Head { .. } | Operation::Encode { .. } => false,
        };
        if retains {
            operation.demand() | crate::batching::Demand::READOUT
        } else {
            operation.demand()
        }
    }

    /// The stores `request`'s scores readouts read, in the order its rows'
    /// readouts name them.
    pub(super) fn retained_rows(&self, request: RequestId) -> RetainedRows<'_> {
        RetainedRows(self.retained.get(&request).map_or(&[], Vec::as_slice))
    }
}

/// The readouts row `row` of `operation` answers, then the retention of its
/// state when `retained` keeps it (which returns no values), as the batch
/// packs them.
pub(super) fn row_readouts(
    operation: &Operation,
    row: usize,
    retained: &RetainedRows<'_>,
) -> Result<Vec<crate::batching::RowReadout>, String> {
    let Operation::Forward {
        position, readouts, ..
    } = operation
    else {
        return Ok(Vec::new());
    };
    let row = position + row;
    let mut packed = readouts
        .iter()
        .filter(|readout| crate::answer_row(readout) == row)
        .map(|readout| match readout {
            Readout {
                kind: ReadoutKind::Logits { tokens },
                ..
            } => Ok(crate::batching::RowReadout::Logits(
                tokens
                    .iter()
                    .map(|token| i32::try_from(token.0).map_err(|_| "token exceeds i32"))
                    .collect::<Result<Vec<_>, _>>()?
                    .into(),
            )),
            Readout {
                kind: ReadoutKind::Head,
                ..
            } => Ok(crate::batching::RowReadout::Head),
            Readout {
                row: query,
                kind: ReadoutKind::Scores { rows },
            } => retained.scores(*query, rows),
        })
        .collect::<Result<Vec<_>, String>>()?;
    // A step stores its retained rows before it reads any scores.
    if let Some((store, index)) = retained.own(row) {
        packed.push(crate::batching::RowReadout::Retain { store, index });
    }
    Ok(packed)
}
