//! Immutable semantic continuation. Successors retain only unconsumed image
//! leases; cloning a checkpoint preserves its own earlier continuation.
use super::{preparation::InputPlan, vision_runtime::Features};
use seismic_runtime::{Buffer, Device};
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

#[derive(Clone)]
pub struct InputState {
    plan: Rc<InputPlan>,
    position: usize,
    width: usize,
    features: HashMap<String, Features>,
}
pub(crate) struct FeatureSlice {
    pub source: Buffer,
    pub destination: usize,
    pub count: usize,
}
pub(crate) struct Assembled {
    pub coordinates: Vec<[i32; 3]>,
    pub features: Vec<FeatureSlice>,
}
impl InputState {
    pub fn new(
        device: &Device,
        plan: Rc<InputPlan>,
        position: usize,
        features: Vec<Features>,
        width: usize,
    ) -> Result<Self, String> {
        if width == 0 || width > i32::MAX as usize || !plan.layout().boundary(position) {
            return Err("invalid conditioned input continuation".into());
        }
        let count = features.len();
        let features: HashMap<_, _> = features
            .into_iter()
            .map(|f| (f.identity().to_owned(), f))
            .collect();
        let required: HashSet<_> = plan
            .layout()
            .spans()
            .iter()
            .filter(|s| s.end > position)
            .map(|s| &s.identity)
            .collect();
        if features.len() != count
            || features.len() != required.len()
            || required.iter().any(|id| !features.contains_key(*id))
        {
            return Err("features must exactly cover unconsumed conditioning".into());
        }
        for span in plan.layout().spans().iter().filter(|s| s.end > position) {
            let feature = &features[&span.identity];
            if feature.rows() != span.end - span.start
                || feature.width() != width
                || !feature.buffer().belongs_to(device)
            {
                return Err("feature geometry or execution owner differs from input".into());
            }
        }
        Ok(Self {
            plan,
            position,
            width,
            features,
        })
    }
    pub fn plan(&self) -> &InputPlan {
        &self.plan
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub(crate) fn belongs_to(&self, device: &Device) -> bool {
        self.features
            .values()
            .all(|f| f.buffer().belongs_to(device))
    }
    /// Retain the successor before publishing it alongside numerical state.
    pub fn after(&self, position: usize) -> Result<Self, String> {
        if position < self.position || !self.plan.layout().boundary(position) {
            return Err("input cannot move backward or cross an illegal boundary".into());
        }
        let remaining: HashSet<_> = self
            .plan
            .layout()
            .spans()
            .iter()
            .filter(|s| s.end > position)
            .map(|s| &s.identity)
            .collect();
        Ok(Self {
            plan: self.plan.clone(),
            position,
            width: self.width,
            features: self
                .features
                .iter()
                .filter(|(id, _)| remaining.contains(id))
                .map(|(id, f)| (id.clone(), f.clone()))
                .collect(),
        })
    }
    pub(crate) fn assemble(&self, tokens: &[u32]) -> Result<Assembled, crate::Error> {
        let end = self
            .position
            .checked_add(tokens.len())
            .filter(|&n| self.plan.layout().boundary(n))
            .ok_or("input advance exceeds position domain")?;
        if tokens.is_empty() || tokens.iter().any(|&t| t > i32::MAX as u32) {
            return Err("input requires nonempty int32 tokens".into());
        }
        let overlap = end
            .min(self.plan.tokens().len())
            .saturating_sub(self.position);
        if tokens[..overlap]
            .iter()
            .zip(self.plan.tokens().iter().skip(self.position))
            .any(|(a, b)| *a != b.0)
        {
            return Err("input tokens differ from the bound prompt".into());
        }
        let mut features = Vec::new();
        let row_bytes = self
            .width
            .checked_mul(4)
            .ok_or("feature row byte overflow")?;
        for span in self
            .plan
            .layout()
            .spans()
            .iter()
            .filter(|s| s.start < end && s.end > self.position)
        {
            let start = self.position.max(span.start);
            let count = end.min(span.end) - start;
            let offset = (start - span.start)
                .checked_mul(row_bytes)
                .ok_or("feature slice overflow")?;
            let length = count
                .checked_mul(row_bytes)
                .ok_or("feature slice overflow")?;
            let feature = self
                .features
                .get(&span.identity)
                .ok_or("unconsumed feature is missing")?;
            features.push(FeatureSlice {
                source: feature
                    .buffer()
                    .view(offset..offset.checked_add(length).ok_or("feature slice overflow")?)?,
                destination: start - self.position,
                count,
            });
        }
        Ok(Assembled {
            coordinates: self.plan.rotary(self.position, tokens.len())?,
            features,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        inputs::{
            media::{DType, PreparedMedia, PreparedTensor},
            TokenId,
        },
        models::qwen35::preparation::{interpret, ImageGeometry},
    };
    fn plan() -> Rc<InputPlan> {
        let geometry = ImageGeometry {
            channels: 1,
            temporal_patch: 1,
            patch: 1,
            merge: 2,
            image_token: TokenId(99),
            start_token: TokenId(98),
            end_token: TokenId(100),
        };
        let media = PreparedMedia::new(
            "a".repeat(64),
            vec![
                PreparedTensor::new("pixel_values".into(), DType::F32, vec![16, 1], vec![0; 64])
                    .unwrap(),
                PreparedTensor::new(
                    "image_grid_thw".into(),
                    DType::I64,
                    vec![1, 3],
                    [1i64, 4, 4]
                        .into_iter()
                        .flat_map(i64::to_le_bytes)
                        .collect(),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        Rc::new(
            interpret(
                [7, 98, 99, 99, 99, 99, 100, 8]
                    .into_iter()
                    .map(TokenId)
                    .collect(),
                &media,
                &"a".repeat(64),
                &geometry,
            )
            .unwrap()
            .plan,
        )
    }
    fn feature(device: &Device, plan: &InputPlan) -> Features {
        Features::test_fixture(
            plan.layout().spans()[0].identity.clone(),
            4,
            3,
            device
                .buffer_from(
                    &(0..12)
                        .flat_map(|n| (n as f32).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
        )
    }
    #[test]
    #[ignore = "requires a Metal device"]
    fn partial_images_retain_exact_slices_and_checkpoint_ownership() {
        let device = Device::metal().unwrap();
        let plan = plan();
        let initial =
            InputState::new(&device, plan.clone(), 0, vec![feature(&device, &plan)], 3).unwrap();
        let checkpoint = initial.clone();
        let first = initial.assemble(&[7, 98, 99, 99]).unwrap();
        assert_eq!(first.features.len(), 1);
        assert_eq!(first.features[0].destination, 2);
        assert_eq!(first.features[0].count, 2);
        assert_eq!(first.features[0].source.len(), 24);
        let partial = initial.after(4).unwrap();
        let second = partial.assemble(&[99, 99, 100]).unwrap();
        assert_eq!(second.features[0].destination, 0);
        let mut actual = vec![0; 24];
        second.features[0].source.read(&mut actual).unwrap();
        assert_eq!(
            actual,
            (6..12)
                .flat_map(|n| (n as f32).to_le_bytes())
                .collect::<Vec<_>>()
        );
        assert_eq!(second.coordinates, plan.rotary(4, 3).unwrap());
        assert!(partial.after(3).is_err());
        let consumed = partial.after(8).unwrap();
        assert!(consumed.features.is_empty());
        assert_eq!(consumed.assemble(&[42]).unwrap().coordinates, vec![[6; 3]]);
        drop((initial, partial, first, second));
        assert_eq!(device.memory_usage().charged, 48);
        assert_eq!(
            checkpoint.assemble(&[7, 98, 99]).unwrap().features[0].count,
            1
        );
        drop(checkpoint);
        assert_eq!(device.memory_usage().charged, 0);
    }
    #[test]
    #[ignore = "requires a Metal device"]
    fn generation_checkpoint_forks_retain_their_own_unconsumed_features() {
        use crate::{
            models::sequence::{OwnedSequence, SequenceWork},
            state::StateStore,
        };
        let device = Rc::new(Device::metal().unwrap());
        let store = StateStore::new(device.clone(), 8, 16, vec![], vec![]).unwrap();
        let plan = plan();
        let input =
            InputState::new(&device, plan.clone(), 0, vec![feature(&device, &plan)], 3).unwrap();
        let sequence = OwnedSequence::with_semantics(store.create().unwrap(), input);
        let checkpoint = sequence.checkpoint_with_semantics().unwrap();
        let mut rows = OwnedSequence::prepare_completed_batch_with_semantics(
            &[SequenceWork {
                sequence: &sequence,
                position: 0,
                count: 6,
            }],
            |states| {
                let (state, input) = &mut states[0];
                let operands = input.assemble(&[7, 98, 99, 99, 99, 99])?;
                assert_eq!(operands.features[0].count, 4);
                let next = input.after(6)?;
                let mut advance = state.begin(6)?;
                // Exercise publication ownership without claiming native parity.
                advance.execute(|_| Ok(()))?;
                advance.commit()?;
                **input = next;
                Ok(vec![Ok(None)])
            },
        )
        .unwrap();
        assert!(sequence.checkpoint_with_semantics().is_err());
        rows[0].commit().unwrap();
        drop(rows);
        let (state, input) = sequence.checkpoint_with_semantics().unwrap();
        assert_eq!((state.position(), input.position()), (6, 6));
        assert!(input.features.is_empty());
        assert_eq!(device.memory_usage().charged, 48);
        let fork = OwnedSequence::with_semantics(checkpoint.0.fork(), checkpoint.1);
        let (state, input) = fork.checkpoint_with_semantics().unwrap();
        assert_eq!((state.position(), input.position()), (0, 0));
        assert_eq!(input.assemble(&[7, 98, 99]).unwrap().features[0].count, 1);
        drop((fork, input));
        assert_eq!(device.memory_usage().charged, 0);
    }
    #[test]
    #[ignore = "requires a Metal device"]
    fn input_rejects_missing_duplicate_wrong_geometry_foreign_and_changed_tokens() {
        let device = Device::metal().unwrap();
        let foreign = Device::metal().unwrap();
        let plan = plan();
        assert!(InputState::new(&device, plan.clone(), 0, vec![], 3).is_err());
        let f = feature(&device, &plan);
        assert!(InputState::new(&device, plan.clone(), 0, vec![f.clone(), f.clone()], 3).is_err());
        assert!(InputState::new(&device, plan.clone(), 0, vec![f.clone()], 4).is_err());
        assert!(InputState::new(&foreign, plan.clone(), 0, vec![f.clone()], 3).is_err());
        assert!(InputState::new(&device, plan.clone(), 6, vec![f.clone()], 3).is_err());
        let input = InputState::new(&device, plan, 0, vec![f], 3).unwrap();
        assert!(input.assemble(&[]).is_err());
        assert!(input.assemble(&[7, 98, 8]).is_err());
        assert!(input.after(usize::MAX).is_err());
        assert_eq!(input.position(), 0);
    }
}
