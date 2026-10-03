use crate::{ModelLoadPlan, WeightPlan};
use magnitude_family_contracts::{ModelDefinition, WeightKind, WeightRole, WeightScope};
use magnitude_state::StageModelView;
use std::{collections::HashSet, fmt, ops::Range, rc::Rc};

/// Refusal of an explicit execution contract, not a placement recommendation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineRefusal {
    Coverage,
    UnsupportedProfile,
    StageCount,
    ForeignDevice,
    ForeignModel,
    InvalidWeightRole(WeightRole),
    Plan(crate::PlanError),
    Memory {
        refusal: crate::ClaimRefusal,
        allocation: u64,
        staged: u64,
    },
    Preparation(String),
}
impl fmt::Display for PipelineRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PipelineRefusal {}

/// One admitted original model and complete ordered coverage supplied by the
/// caller. The representation is N-shaped; this does not qualify N-device runs.
#[derive(Clone)]
pub struct PipelineModel {
    pub(super) definition: Rc<ModelDefinition>,
    pub(super) ranges: Vec<Range<usize>>,
}
impl PipelineModel {
    pub fn new(
        definition: Rc<ModelDefinition>,
        ranges: Vec<Range<usize>>,
    ) -> Result<Self, PipelineRefusal> {
        definition
            .validate()
            .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?;
        let mut next = 0;
        for range in &ranges {
            if range.start != next
                || StageModelView::new(&definition.decoder, range.clone()).is_err()
            {
                return Err(PipelineRefusal::Coverage);
            }
            next = range.end;
        }
        if ranges.is_empty() || next != definition.decoder.blocks.len() {
            return Err(PipelineRefusal::Coverage);
        }
        Ok(Self { definition, ranges })
    }
    pub fn stages(&self) -> impl Iterator<Item = StageAssignment> + '_ {
        self.ranges.iter().cloned().map(|range| StageAssignment {
            definition: self.definition.clone(),
            range,
        })
    }
    /// Project only whole-block pipeline intent. Shards and empty groups are
    /// refused before device opening; representability is not qualification.
    pub fn from_placement(
        definition: Rc<ModelDefinition>,
        placement: &crate::placement::ModelPlacement,
    ) -> Result<(Self, Vec<seismic::DeviceSelector>), PipelineRefusal> {
        use crate::placement::{ModelRegion, PartitionAssignment};
        let assigned = placement
            .groups
            .iter()
            .map(|group| match group.partitions.as_slice() {
                [PartitionAssignment {
                    device,
                    region: ModelRegion::DecoderBlocks(range),
                }] => Ok((range.clone(), *device)),
                _ => Err(PipelineRefusal::UnsupportedProfile),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (ranges, devices) = assigned.into_iter().unzip();
        Ok((Self::new(definition, ranges)?, devices))
    }
}

/// An original decoder view plus derived entry/exit responsibility. Callers
/// cannot independently set endpoint flags or substitute a renumbered decoder.
#[derive(Clone)]
pub struct StageAssignment {
    definition: Rc<ModelDefinition>,
    range: Range<usize>,
}
impl StageAssignment {
    pub(crate) fn original_model(&self) -> &Rc<ModelDefinition> {
        &self.definition
    }
    pub fn definition(&self) -> &ModelDefinition {
        &self.definition
    }
    pub fn view(&self) -> StageModelView<'_> {
        StageModelView::new(&self.definition.decoder, self.range.clone())
            .expect("validated original range")
    }
    pub fn owns_entry(&self) -> bool {
        self.range.start == 0
    }
    pub fn owns_readout(&self) -> bool {
        self.range.end == self.definition.decoder.blocks.len()
    }
    pub fn owns_role(&self, role: WeightRole) -> Result<bool, PipelineRefusal> {
        match role.scope {
            WeightScope::TargetSublayer(index)
            | WeightScope::TargetBranch {
                sublayer: index, ..
            } => {
                if index.block as usize >= self.definition.decoder.blocks.len() {
                    return Err(PipelineRefusal::InvalidWeightRole(role));
                }
                Ok(self.range.contains(&(index.block as usize)))
            }
            WeightScope::Target => match role.kind {
                WeightKind::Embedding => Ok(self.owns_entry()),
                WeightKind::OutputNorm | WeightKind::Output => Ok(self.owns_readout()),
                _ => Err(PipelineRefusal::InvalidWeightRole(role)),
            },
            _ => Err(PipelineRefusal::InvalidWeightRole(role)),
        }
    }
    /// Visit only original semantic roles assigned to this stage. Storage
    /// deduplication is local to a device: tied entry/output on different stages
    /// contribute a physical allocation to each device's separate budget.
    pub fn weights<'a>(
        &self,
        load: &'a ModelLoadPlan,
    ) -> Result<Vec<&'a WeightPlan>, PipelineRefusal> {
        if load.head().is_some() || load.vision().is_some() || !load.host_tables().is_empty() {
            return Err(PipelineRefusal::UnsupportedProfile);
        }
        let d = &self.definition.decoder;
        let expected = std::iter::once((
            WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::Embedding,
            },
            &d.entry.embedding,
        ))
        .chain(crate::operators::decoder_weights(d))
        .chain([
            (
                WeightRole {
                    scope: WeightScope::Target,
                    kind: WeightKind::OutputNorm,
                },
                d.exit.norm.weight(),
            ),
            (
                WeightRole {
                    scope: WeightScope::Target,
                    kind: WeightKind::Output,
                },
                &d.exit.output,
            ),
        ])
        .collect::<std::collections::HashMap<_, _>>();
        if load.weights().count() != expected.len() {
            return Err(PipelineRefusal::ForeignModel);
        }
        let mut assigned = Vec::new();
        for weight in load.weights() {
            if expected
                .get(&weight.role)
                .is_none_or(|descriptor| **descriptor != weight.descriptor)
            {
                return Err(PipelineRefusal::ForeignModel);
            }
            if weight.component.identity != self.definition.artifact_identity.target {
                return Err(PipelineRefusal::ForeignModel);
            }
            if self.owns_role(weight.role)? {
                assigned.push(weight);
            }
        }
        Ok(assigned)
    }
    pub fn resident_bytes(&self, load: &ModelLoadPlan) -> Result<u64, PipelineRefusal> {
        let mut seen = HashSet::new();
        self.weights(load)?.iter().try_fold(0u64, |bytes, weight| {
            if !seen.insert(weight.storage_identity()) {
                return Ok(bytes);
            }
            bytes
                .checked_add(
                    weight
                        .storage_bytes()
                        .map_err(PipelineRefusal::Preparation)?,
                )
                .ok_or_else(|| PipelineRefusal::Preparation("stage weight charge overflows".into()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_family_contracts::SublayerIndex;
    fn definition() -> Rc<ModelDefinition> {
        let mut d = crate::planning::tests::fixture_definition();
        d.decoder.blocks = vec![d.decoder.blocks[0].clone(); 4];
        Rc::new(d)
    }
    #[test]
    fn complete_coverage_is_ordered_and_n_shaped_but_runtime_is_binary() {
        let d = definition();
        for ranges in [
            std::iter::once(0..4).collect(),
            vec![0..1, 1..4],
            vec![0..1, 1..2, 2..4],
        ] {
            let model = PipelineModel::new(d.clone(), ranges.clone()).unwrap();
            assert_eq!(
                super::super::TwoStageCudaPipeline::qualify(
                    &model,
                    &[
                        seismic::DeviceSelector::Cuda { uuid: [1; 16] },
                        seismic::DeviceSelector::Cuda { uuid: [2; 16] }
                    ]
                )
                .is_ok(),
                ranges.len() == 2
            );
            for (assignment, range) in model.stages().zip(ranges) {
                assert_eq!(assignment.view().global_range(), range);
                assert!(std::ptr::eq(assignment.view().decoder(), &d.decoder));
            }
        }
        for ranges in [
            vec![],
            std::iter::once(0..2).collect(),
            std::iter::once(1..4).collect(),
            vec![0..2, 3..4],
            vec![0..3, 2..4],
            vec![0..2, 2..2, 2..4],
            std::iter::once(0..5).collect(),
        ] {
            assert!(matches!(
                PipelineModel::new(d.clone(), ranges),
                Err(PipelineRefusal::Coverage)
            ));
        }
    }
    #[test]
    fn tied_weight_storage_is_charged_separately_per_assigned_device() {
        let mut d = (*definition()).clone();
        d.decoder.exit.output = d.decoder.entry.embedding.clone();
        let manifest = crate::planning::tests::fixture_manifest(&d);
        let load = ModelLoadPlan::derive(
            &manifest,
            &d,
            crate::ComponentSelection {
                head: false,
                vision: false,
            },
            seismic::Layout::Packet,
        )
        .unwrap();
        let model = PipelineModel::new(Rc::new(d), vec![0..2, 2..4]).unwrap();
        let stages = model.stages().collect::<Vec<_>>();
        let endpoint = |stage: &StageAssignment, kind| {
            stage
                .weights(&load)
                .unwrap()
                .into_iter()
                .find(|w| {
                    w.role
                        == WeightRole {
                            scope: WeightScope::Target,
                            kind,
                        }
                })
                .unwrap()
        };
        let embedding = endpoint(&stages[0], WeightKind::Embedding);
        let output = endpoint(&stages[1], WeightKind::Output);
        assert_eq!(embedding.storage_identity(), output.storage_identity());
        for stage in &stages {
            let mut unique = std::collections::HashMap::new();
            for w in stage.weights(&load).unwrap() {
                unique.insert(w.storage_identity(), w.storage_bytes().unwrap());
            }
            assert_eq!(
                stage.resident_bytes(&load).unwrap(),
                unique.values().sum::<u64>()
            );
        }
        // This extra copy is not elided by a global cross-device cache.
        let mut unique = std::collections::HashMap::new();
        for w in load.weights() {
            unique.insert(w.storage_identity(), w.storage_bytes().unwrap());
        }
        // Block fixture tensors are tied too; count the intersection of local
        // storage identities, not only the shared embedding/readout.
        let first = stages[0]
            .weights(&load)
            .unwrap()
            .into_iter()
            .map(|w| (w.storage_identity(), w.storage_bytes().unwrap()))
            .collect::<std::collections::HashMap<_, _>>();
        let last = stages[1]
            .weights(&load)
            .unwrap()
            .into_iter()
            .map(|w| (w.storage_identity(), w.storage_bytes().unwrap()))
            .collect::<std::collections::HashMap<_, _>>();
        let duplicated = first
            .iter()
            .filter(|(key, _)| last.contains_key(*key))
            .map(|(_, bytes)| bytes)
            .sum::<u64>();
        assert!(duplicated >= embedding.storage_bytes().unwrap());
        assert_eq!(
            stages
                .iter()
                .map(|s| s.resident_bytes(&load).unwrap())
                .sum::<u64>(),
            unique.values().sum::<u64>() + duplicated
        );
    }

    #[test]
    fn assigned_roles_keep_global_ids_and_endpoint_ownership() {
        let model = PipelineModel::new(definition(), vec![0..2, 2..4]).unwrap();
        let stages = model.stages().collect::<Vec<_>>();
        for block in 0..4 {
            for scope in [
                WeightScope::TargetSublayer(SublayerIndex { block, sublayer: 0 }),
                WeightScope::TargetBranch {
                    sublayer: SublayerIndex { block, sublayer: 1 },
                    branch: 0,
                },
            ] {
                let role = WeightRole {
                    scope,
                    kind: WeightKind::InputNorm,
                };
                assert_eq!(stages[0].owns_role(role).unwrap(), block < 2);
                assert_eq!(stages[1].owns_role(role).unwrap(), block >= 2);
            }
        }
        for (kind, owner) in [
            (WeightKind::Embedding, 0),
            (WeightKind::OutputNorm, 1),
            (WeightKind::Output, 1),
        ] {
            let role = WeightRole {
                scope: WeightScope::Target,
                kind,
            };
            for (i, stage) in stages.iter().enumerate() {
                assert_eq!(stage.owns_role(role).unwrap(), i == owner);
            }
        }
        let unsupported = WeightRole {
            scope: WeightScope::Target,
            kind: WeightKind::PerLayerTable,
        };
        assert!(stages[0].owns_role(unsupported).is_err());
    }
}
