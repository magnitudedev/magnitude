//! State maintenance owns physical state until a completed submission is reconciled.

use crate::{InvariantError, NativeGraphWorkspaceLease, ResourceDomainId};
use magnitude_batching::{StateBatchKind, ValidatedStateBatch};
use magnitude_state::{OwnedCodecAdvance, StateStore, StoreCopy};
use std::rc::Rc;

pub enum StateWork {
    StoreCopy(StoreCopy),
    CodecConversion(OwnedCodecAdvance),
}

pub struct StateLaunchInputs {
    batch: ValidatedStateBatch,
    work: StateWork,
    graph_workspace: NativeGraphWorkspaceLease,
}

impl StateLaunchInputs {
    pub fn new(
        batch: ValidatedStateBatch,
        work: StateWork,
        graph_workspace: NativeGraphWorkspaceLease,
    ) -> Self {
        Self {
            batch,
            work,
            graph_workspace,
        }
    }

    fn validate(
        &self,
        source_store: &Rc<StateStore>,
        destination_store: Option<&Rc<StateStore>>,
        domain: &ResourceDomainId,
    ) -> Result<(), InvariantError> {
        let invalid = |detail: String| InvariantError {
            context: "state launch",
            detail,
        };
        if self.graph_workspace.domain() != domain {
            return Err(invalid(
                "state lease differs from the planned domain/class".into(),
            ));
        }
        match (&self.work, self.batch.kind()) {
            (StateWork::StoreCopy(work), StateBatchKind::Copy) => {
                if !work.belongs_to(source_store)
                    || work.rows() != self.batch.actual_rows()
                    || self.batch.copies() != Some(work.copies())
                {
                    return Err(invalid(
                        "store copy controls differ from the held slabs".into(),
                    ));
                }
            }
            (
                StateWork::CodecConversion(work),
                StateBatchKind::CodecConversion {
                    source,
                    destination,
                },
            ) => {
                if !work.source_belongs_to(source_store)
                    || !destination_store.is_some_and(|store| work.destination_belongs_to(store))
                    || work.source_codec() != source
                    || work.destination_codec() != destination
                    || work.rows() != self.batch.actual_rows()
                    || self.batch.conversions() != Some(work.conversions())
                {
                    return Err(invalid(
                        "codec controls differ from the owned source/destination transaction"
                            .into(),
                    ));
                }
            }
            _ => {
                return Err(invalid(
                    "state batch and owned maintenance operation differ".into(),
                ));
            }
        }
        Ok(())
    }
}

pub struct ValidatedStateLaunch {
    core: StateLaunchCore,
    graph_workspace: NativeGraphWorkspaceLease,
}

impl ValidatedStateLaunch {
    pub fn new(
        inputs: StateLaunchInputs,
        source_store: &Rc<StateStore>,
        destination_store: Option<&Rc<StateStore>>,
        domain: &ResourceDomainId,
    ) -> Result<Self, (StateLaunchInputs, InvariantError)> {
        if let Err(error) = inputs.validate(source_store, destination_store, domain) {
            return Err((inputs, error));
        }
        let StateLaunchInputs {
            batch,
            work,
            graph_workspace,
        } = inputs;
        Ok(Self {
            core: StateLaunchCore {
                batch,
                work,
                domain: domain.clone(),
            },
            graph_workspace,
        })
    }

    pub fn domain(&self) -> &ResourceDomainId {
        &self.core.domain
    }
    pub(crate) fn execution_parts_mut(
        &mut self,
    ) -> (
        &ValidatedStateBatch,
        &mut StateWork,
        &mut NativeGraphWorkspaceLease,
    ) {
        (
            &self.core.batch,
            &mut self.core.work,
            &mut self.graph_workspace,
        )
    }
    pub(crate) fn into_submission_parts(self) -> (StateLaunchCore, NativeGraphWorkspaceLease) {
        (self.core, self.graph_workspace)
    }
}

pub struct StateLaunchCore {
    batch: ValidatedStateBatch,
    work: StateWork,
    domain: ResourceDomainId,
}

impl StateLaunchCore {
    pub fn batch(&self) -> &ValidatedStateBatch {
        &self.batch
    }
    pub fn work(&self) -> &StateWork {
        &self.work
    }
    pub fn into_parts(self) -> (ValidatedStateBatch, StateWork) {
        (self.batch, self.work)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_batching::StateBatchError;
    use magnitude_state::{Codec, CodecConversionStep, HistoryDomainId, KvCodec, LayerRef, VectorKind};
    use seismic::DType;

    fn step(
        domain: usize,
        layer: u32,
        vector: VectorKind,
        source_planes: Vec<usize>,
        destination_planes: Vec<usize>,
        from: Vec<usize>,
        to: Vec<usize>,
    ) -> CodecConversionStep {
        CodecConversionStep {
            domain: HistoryDomainId(domain),
            layer: LayerRef::Target(layer),
            vector,
            source_codec: Codec::Dense { dtype: DType::F16 },
            destination_codec: Codec::Dense { dtype: DType::F16 },
            source_planes,
            destination_planes,
            from,
            to,
        }
    }

    #[test]
    fn multi_domain_codec_conversion_sums_per_domain_rows_and_checks_per_domain_capacities() {
        let steps = vec![
            step(0, 0, VectorKind::Key, vec![0], vec![0], vec![0, 1, 2], vec![4, 5, 6]),
            step(0, 0, VectorKind::Value, vec![1], vec![1], vec![0, 1, 2], vec![4, 5, 6]),
            step(1, 1, VectorKind::Key, vec![2], vec![2], vec![10, 11], vec![0, 1]),
            step(1, 1, VectorKind::Value, vec![3], vec![3], vec![10, 11], vec![0, 1]),
        ];
        // Domain 0 has 3 rows, domain 1 has 2 rows -> actual_rows == 5.
        let batch = ValidatedStateBatch::convert(
            KvCodec::Dense,
            KvCodec::AffineK8V4,
            steps.clone(),
            &[8, 16],
            &[8, 4],
            8,
        )
        .unwrap();
        assert_eq!(batch.actual_rows(), 5);

        // Exceeding domain 1's destination capacity (1) rejects even though domain 0 has capacity 8.
        assert_eq!(
            ValidatedStateBatch::convert(
                KvCodec::Dense,
                KvCodec::AffineK8V4,
                steps.clone(),
                &[8, 16],
                &[8, 1],
                8,
            )
            .unwrap_err(),
            StateBatchError::InvalidMapping
        );

        // Class capacity is checked against the sum across domains (5 > 4).
        assert_eq!(
            ValidatedStateBatch::convert(
                KvCodec::Dense,
                KvCodec::AffineK8V4,
                steps.clone(),
                &[8, 16],
                &[8, 4],
                4,
            )
            .unwrap_err(),
            StateBatchError::Capacity {
                required: 5,
                available: 4,
            }
        );

        // Mismatched row mapping within the same domain is rejected.
        let mut mismatched = steps;
        mismatched[1].to[0] = 7;
        assert_eq!(
            ValidatedStateBatch::convert(
                KvCodec::Dense,
                KvCodec::AffineK8V4,
                mismatched,
                &[8, 16],
                &[8, 4],
                8,
            )
            .unwrap_err(),
            StateBatchError::InvalidMapping
        );
    }
}
