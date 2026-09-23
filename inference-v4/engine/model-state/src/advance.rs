//! Owned tentative state, movable through an in-flight program submission.

use super::{
    install_commit, split_extent_prefix, BankHandle, Codec, ComponentDescriptor, Error, Extent,
    KvCodec, LayerRef, PlaneBuffer, PlaneCopy, SequenceState, StateStore, VectorKind,
    MAX_VISIBLE_SEGMENTS,
};
use seismic::Tensor;
use std::rc::Rc;

/// A read-only view whose tensors and row claims are owned by the transaction.
#[derive(Clone, Copy)]
pub struct OwnedAdvanceBindings<'a> {
    pub previous: &'a [Tensor],
    pub following: &'a [Tensor],
    pub previous_bank: usize,
    pub following_bank: usize,
    pub history: &'a [PlaneBuffer],
    pub destinations: &'a [usize],
}

/// Tentative successor state. Moving this value into a launch and submission
/// preserves the source state, reserved extents, recurrent bank, and bindings
/// until physical completion and reconciliation.
pub struct OwnedStateAdvance {
    state: SequenceState,
    count: usize,
    extents: Vec<Rc<Extent>>,
    following: BankHandle,
    history: Vec<PlaneBuffer>,
    destinations: Vec<usize>,
}

/// Preparation keeps ownership of the source sequence in every outcome.
/// Deferral is a capacity result, not a partially reserved transaction.
pub enum OwnedCompactionPreparation {
    NotNeeded(SequenceState),
    Ready(OwnedCompaction),
    Deferred {
        state: SequenceState,
        segments: usize,
        visible_rows: usize,
    },
}

/// Tentative history copy with a reserved contiguous destination. Its source
/// extents remain accepted until the state program completes and commits it.
pub struct OwnedCompaction {
    state: SequenceState,
    destination: Rc<Extent>,
    copies: Vec<PlaneCopy>,
    history: Vec<PlaneBuffer>,
}

/// Physical copy bindings pinned by the owned compaction until completion.
#[derive(Clone, Copy)]
pub struct OwnedCompactionBindings<'a> {
    pub history: &'a [PlaneBuffer],
    pub copies: &'a [PlaneCopy],
}

/// One semantic vector conversion. A codec can use several physical planes;
/// their indices are carried together rather than mistaken for one copy plane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecConversionStep {
    pub layer: LayerRef,
    pub vector: VectorKind,
    pub source_codec: Codec,
    pub destination_codec: Codec,
    pub source_planes: Vec<usize>,
    pub destination_planes: Vec<usize>,
    pub from: Vec<usize>,
    pub to: Vec<usize>,
}

/// Physical buffers remain pinned by the owned transaction through finish.
pub struct OwnedCodecBindings<'a> {
    pub source_history: &'a [PlaneBuffer],
    pub destination_history: &'a [PlaneBuffer],
    pub source_recurrent: &'a [Tensor],
    pub destination_recurrent: &'a [Tensor],
    pub conversions: &'a [CodecConversionStep],
}

/// A reserved destination and a complete, checked conversion program. Nothing
/// about the destination becomes accepted until a completed submission commits.
pub struct OwnedCodecAdvance {
    source: SequenceState,
    destination: SequenceState,
    extents: Vec<Rc<Extent>>,
    source_history: Vec<PlaneBuffer>,
    destination_history: Vec<PlaneBuffer>,
    conversions: Vec<CodecConversionStep>,
    source_codec: KvCodec,
    destination_codec: KvCodec,
    rows: usize,
}

impl OwnedCodecAdvance {
    pub fn begin(
        source: SequenceState,
        destination: SequenceState,
    ) -> Result<Self, (SequenceState, SequenceState, Error)> {
        match Self::prepare(&source, &destination) {
            Ok((source_codec, destination_codec, from, rows)) => {
                let extents = match destination.store.reserve(rows) {
                    Ok(extents) => extents,
                    Err(error) => return Err((source, destination, error)),
                };
                let to = extents
                    .iter()
                    .flat_map(|extent| extent.start..extent.start + extent.count)
                    .collect::<Vec<_>>();
                let source_history = match source.store.history_planes() {
                    Ok(planes) => planes,
                    Err(error) => return Err((source, destination, error)),
                };
                let destination_history = match destination.store.history_planes() {
                    Ok(planes) => planes,
                    Err(error) => return Err((source, destination, error)),
                };
                let conversions = conversion_steps(&source, &destination, &from, &to);
                Ok(Self {
                    source,
                    destination,
                    extents,
                    source_history,
                    destination_history,
                    conversions,
                    source_codec,
                    destination_codec,
                    rows,
                })
            }
            Err(error) => Err((source, destination, error)),
        }
    }

    fn prepare(
        source: &SequenceState,
        destination: &SequenceState,
    ) -> Result<(KvCodec, KvCodec, Vec<usize>, usize), Error> {
        if Rc::ptr_eq(&source.store, &destination.store)
            || !Rc::ptr_eq(&source.store.device, &destination.store.device)
            || destination.position != 0
            || !destination.extents.is_empty()
            || destination.history_start != 0
            || destination.retained_start != 0
            || source.store.component_specs != destination.store.component_specs
            || source.position > destination.store.context_capacity
            || source.expected_end > destination.store.context_capacity
        {
            return Err(Error::Request(
                "codec conversion requires a fresh compatible destination on the same device"
                    .into(),
            ));
        }
        let source_components = &source.store.components;
        let destination_components = &destination.store.components;
        if source_components.is_empty() || source_components.len() != destination_components.len() {
            return Err(Error::Request(
                "codec conversion requires matching nonempty history layouts".into(),
            ));
        }
        let source_codec = identify_codec(source_components)?;
        let destination_codec = identify_codec(destination_components)?;
        if source_codec == destination_codec {
            return Err(Error::Request(
                "codec conversion requires distinct codecs".into(),
            ));
        }
        for (before, after) in source_components.iter().zip(destination_components) {
            if before.layer != after.layer
                || before.codec.key_width != after.codec.key_width
                || before.codec.value_width != after.codec.value_width
            {
                return Err(Error::Request(
                    "codec conversion layers or vector widths differ".into(),
                ));
            }
        }
        let from = source
            .history_ranges()
            .into_iter()
            .flat_map(|(start, count)| start..start + count)
            .collect::<Vec<_>>();
        if from.is_empty() {
            return Err(Error::Request(
                "codec conversion requires visible history rows".into(),
            ));
        }
        let rows = from.len();
        if rows > destination.store.history_capacity {
            return Err(Error::Request(
                "codec conversion exceeds destination history capacity".into(),
            ));
        }
        Ok((source_codec, destination_codec, from, rows))
    }

    pub fn source_belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.source.belongs_to(store)
    }
    pub fn destination_belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.destination.belongs_to(store)
    }
    pub fn source_codec(&self) -> KvCodec {
        self.source_codec
    }
    pub fn destination_codec(&self) -> KvCodec {
        self.destination_codec
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn conversions(&self) -> &[CodecConversionStep] {
        &self.conversions
    }
    pub fn bindings(&self) -> OwnedCodecBindings<'_> {
        OwnedCodecBindings {
            source_history: &self.source_history,
            destination_history: &self.destination_history,
            source_recurrent: self.source.bank.values(),
            destination_recurrent: self.destination.bank.values(),
            conversions: &self.conversions,
        }
    }
    pub fn abort(self) -> (SequenceState, SequenceState) {
        (self.source, self.destination)
    }
    pub fn commit(mut self) -> SequenceState {
        self.destination.position = self.source.position;
        self.destination.expected_end = self.source.expected_end;
        self.destination.history_start = self.source.history_start;
        self.destination.retained_start = self.source.history_start;
        self.destination.extents = std::mem::take(&mut self.extents);
        self.destination
    }
}

fn identify_codec(components: &[ComponentDescriptor]) -> Result<KvCodec, Error> {
    for candidate in [KvCodec::Dense, KvCodec::AffineK8V4, KvCodec::RotatedK4V4] {
        if components.iter().all(|component| {
            let dense_dtype = match component.codec.key {
                Codec::Dense { dtype } => dtype,
                _ => seismic::DType::F16,
            };
            let expected = candidate.spec(
                dense_dtype,
                component.codec.key_width,
                component.codec.value_width,
            );
            component.codec == expected
        }) {
            return Ok(candidate);
        }
    }
    Err(Error::Request(
        "history layout is not one named codec policy".into(),
    ))
}

fn conversion_steps(
    source: &SequenceState,
    destination: &SequenceState,
    from: &[usize],
    to: &[usize],
) -> Vec<CodecConversionStep> {
    let mut steps = Vec::with_capacity(source.store.components.len() * 2);
    let mut source_base = 0;
    let mut destination_base = 0;
    for (before, after) in source
        .store
        .components
        .iter()
        .zip(&destination.store.components)
    {
        for vector in [VectorKind::Key, VectorKind::Value] {
            let source_planes = before
                .planes()
                .iter()
                .enumerate()
                .filter_map(|(index, plane)| {
                    (plane.vector == vector).then_some(source_base + index)
                })
                .collect();
            let destination_planes = after
                .planes()
                .iter()
                .enumerate()
                .filter_map(|(index, plane)| {
                    (plane.vector == vector).then_some(destination_base + index)
                })
                .collect();
            steps.push(CodecConversionStep {
                layer: before.layer,
                vector,
                source_codec: if vector == VectorKind::Key {
                    before.codec.key
                } else {
                    before.codec.value
                },
                destination_codec: if vector == VectorKind::Key {
                    after.codec.key
                } else {
                    after.codec.value
                },
                source_planes,
                destination_planes,
                from: from.to_vec(),
                to: to.to_vec(),
            });
        }
        source_base += before.planes().len();
        destination_base += after.planes().len();
    }
    steps
}

impl OwnedCompaction {
    pub fn prepare(
        state: SequenceState,
    ) -> Result<OwnedCompactionPreparation, (SequenceState, Error)> {
        let ranges = state.history_ranges();
        if ranges.len() <= MAX_VISIBLE_SEGMENTS {
            return Ok(OwnedCompactionPreparation::NotNeeded(state));
        }
        let Some(visible_rows) = ranges
            .iter()
            .try_fold(0usize, |total, (_, count)| total.checked_add(*count))
        else {
            return Err((
                state,
                Error::Request("visible history row count overflow".into()),
            ));
        };
        let Some(destination) = state.store.reserve_contiguous(visible_rows) else {
            return Ok(OwnedCompactionPreparation::Deferred {
                state,
                segments: ranges.len(),
                visible_rows,
            });
        };
        let history = match state.store.history_planes() {
            Ok(history) => history,
            Err(error) => return Err((state, error)),
        };
        let from = ranges
            .iter()
            .flat_map(|(start, count)| *start..*start + *count)
            .collect::<Vec<_>>();
        let to = (destination.start..destination.start + destination.count).collect::<Vec<_>>();
        let copies = state
            .store
            .components
            .iter()
            .flat_map(ComponentDescriptor::planes)
            .enumerate()
            .map(|(plane_index, _)| PlaneCopy {
                plane_index,
                from: from.clone(),
                to: to.clone(),
            })
            .collect();
        Ok(OwnedCompactionPreparation::Ready(Self {
            state,
            destination,
            copies,
            history,
        }))
    }

    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.state.belongs_to(store)
    }
    pub fn rows(&self) -> usize {
        self.destination.count
    }
    pub fn copies(&self) -> &[PlaneCopy] {
        &self.copies
    }
    pub fn bindings(&self) -> OwnedCompactionBindings<'_> {
        OwnedCompactionBindings {
            history: &self.history,
            copies: &self.copies,
        }
    }

    /// Called only after the owning state submission has physically finished.
    pub fn commit(mut self) -> SequenceState {
        self.state.extents = vec![self.destination];
        self.state.retained_start = self.state.history_start;
        self.state
    }

    pub fn abort(self) -> SequenceState {
        self.state
    }
}

impl OwnedStateAdvance {
    /// Returns ownership of the unchanged source state if reservation fails.
    pub fn begin(state: SequenceState, count: usize) -> Result<Self, (SequenceState, Error)> {
        if count == 0 || count > state.store.context_capacity.saturating_sub(state.position) {
            return Err((state, Error::from("advance exceeds context capacity")));
        }
        let extents = match state.store.reserve(count) {
            Ok(extents) => extents,
            Err(error) => return Err((state, error)),
        };
        let following = match state.store.banks.acquire() {
            Ok(following) => following,
            Err(error) => return Err((state, error)),
        };
        let history = match state.store.history_planes() {
            Ok(history) => history,
            Err(error) => return Err((state, error)),
        };
        let destinations = extents
            .iter()
            .flat_map(|extent| extent.start..extent.start + extent.count)
            .collect();
        Ok(Self {
            state,
            count,
            extents,
            following,
            history,
            destinations,
        })
    }

    pub fn position(&self) -> usize {
        self.state.position
    }

    pub fn rows(&self) -> usize {
        self.count
    }

    /// A launch must join every advance with the exact state arena selected
    /// by its executor, not merely an arena on the same device.
    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.state.belongs_to(store)
    }

    pub fn history_ranges(&self) -> Vec<(usize, usize)> {
        self.state.history_ranges()
    }

    pub fn bindings(&self) -> OwnedAdvanceBindings<'_> {
        OwnedAdvanceBindings {
            previous: self.state.bank.values(),
            following: self.following.values(),
            previous_bank: self.state.bank.index(),
            following_bank: self.following.index(),
            history: &self.history,
            destinations: &self.destinations,
        }
    }

    /// Abort a submission that did not complete successfully and recover its
    /// unchanged accepted state. Dropping the transaction also releases every
    /// tentative claim, for terminal teardown that needs no source recovery.
    pub fn abort(self) -> SequenceState {
        self.state
    }

    pub fn commit_all(self) -> Result<OwnedAdvanceResolution, (SequenceState, Error)> {
        let rows = self.count;
        self.commit(rows)
    }

    /// Publish exactly the accepted physical prefix. A recurrent interior
    /// prefix yields owned repair work before any successor becomes visible.
    pub fn commit(self, accepted: usize) -> Result<OwnedAdvanceResolution, (SequenceState, Error)> {
        let Self {
            mut state,
            count,
            mut extents,
            mut following,
            history: _,
            destinations: _,
        } = self;
        if accepted > count {
            return Err((
                state,
                Error::from("accepted prefix exceeds advance row count"),
            ));
        }
        if accepted == 0 {
            return Ok(OwnedAdvanceResolution::Aborted(state));
        }
        if accepted == count {
            install_commit(&mut state, &mut extents, &mut following, count);
            return Ok(OwnedAdvanceResolution::Committed(state));
        }
        let (mut kept, released) = match split_extent_prefix(extents, accepted) {
            Ok(parts) => parts,
            Err(error) => return Err((state, Error::from(error))),
        };
        drop(released);
        if following.values().is_empty() {
            install_commit(&mut state, &mut kept, &mut following, accepted);
            return Ok(OwnedAdvanceResolution::Committed(state));
        }
        Ok(OwnedAdvanceResolution::Repair(OwnedRepairAdvance {
            state,
            accepted,
            extents: kept,
            following,
        }))
    }
}

pub enum OwnedAdvanceResolution {
    Aborted(SequenceState),
    Committed(SequenceState),
    Repair(OwnedRepairAdvance),
}

/// Recompute the recurrent successor of an accepted interior prefix using an
/// ordinary state-program submission, then publish it atomically.
pub struct OwnedRepairAdvance {
    state: SequenceState,
    accepted: usize,
    extents: Vec<Rc<Extent>>,
    following: BankHandle,
}

impl OwnedRepairAdvance {
    pub fn belongs_to(&self, store: &Rc<StateStore>) -> bool {
        self.state.belongs_to(store)
    }

    pub fn rows(&self) -> usize {
        self.accepted
    }

    pub fn history_ranges(&self) -> Vec<(usize, usize)> {
        self.state.history_ranges()
    }

    pub fn previous_bank(&self) -> usize {
        self.state.bank.index()
    }

    pub fn previous(&self) -> &[Tensor] {
        self.state.bank.values()
    }

    pub fn following(&self) -> &[Tensor] {
        self.following.values()
    }

    pub fn commit(mut self) -> SequenceState {
        install_commit(
            &mut self.state,
            &mut self.extents,
            &mut self.following,
            self.accepted,
        );
        self.state
    }

    pub fn abort(self) -> SequenceState {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BankCapacity, CodecSpec, ComponentDescriptor, ComponentSpec, LayerRef, StateStore,
    };
    use seismic::{BackendName, DType, DeviceCatalog};

    #[test]
    fn abort_recovers_source_and_releases_tentative_capacity() {
        let Some(device) = DeviceCatalog::discover()
            .ok()
            .and_then(|catalog| catalog.open_backend(BackendName::Cpu).ok())
        else {
            return;
        };
        let store = StateStore::new(
            Rc::new(device),
            4,
            4,
            vec![
                ComponentDescriptor::new(LayerRef::Target(0), CodecSpec::dense(DType::F32, 1, 1))
                    .unwrap(),
            ],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let state = store.create().unwrap();
        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        assert_eq!(advance.bindings().destinations.len(), 2);
        assert_eq!(store.occupied_rows(), 2);
        let state = advance.abort();
        assert_eq!(state.position(), 0);
        assert_eq!(store.occupied_rows(), 0);

        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let OwnedAdvanceResolution::Committed(state) = advance.commit_all().ok().unwrap() else {
            panic!("full accepted prefix must commit");
        };
        assert_eq!(state.position(), 2);
        assert_eq!(store.occupied_rows(), 2);
    }

    #[test]
    fn codec_conversion_reserves_and_publishes_only_after_completion() {
        let Some(device) = DeviceCatalog::discover()
            .ok()
            .and_then(|catalog| catalog.open_backend(BackendName::Cpu).ok())
        else {
            return;
        };
        let device = Rc::new(device);
        let source_store = StateStore::new(
            device.clone(),
            4,
            4,
            vec![
                ComponentDescriptor::new(LayerRef::Target(0), CodecSpec::dense(DType::F16, 8, 8))
                    .unwrap(),
            ],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let destination_store = StateStore::new(
            device,
            4,
            4,
            vec![ComponentDescriptor::new(
                LayerRef::Target(0),
                KvCodec::AffineK8V4.spec(DType::F16, 8, 8),
            )
            .unwrap()],
            vec![],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let source = OwnedStateAdvance::begin(source_store.create().unwrap(), 2)
            .ok()
            .unwrap();
        let OwnedAdvanceResolution::Committed(source) = source.commit_all().ok().unwrap() else {
            panic!("source rows must commit");
        };
        let destination = destination_store.create().unwrap();
        let conversion = OwnedCodecAdvance::begin(source, destination).ok().unwrap();
        assert_eq!(conversion.rows(), 2);
        assert_eq!(conversion.source_codec(), KvCodec::Dense);
        assert_eq!(conversion.destination_codec(), KvCodec::AffineK8V4);
        assert_eq!(conversion.conversions().len(), 2);
        assert_eq!(conversion.bindings().destination_history.len(), 6);
        assert_eq!(destination_store.occupied_rows(), 2);
        let (source, destination) = conversion.abort();
        assert_eq!(source.position(), 2);
        assert_eq!(destination.position(), 0);
        assert_eq!(destination_store.occupied_rows(), 0);
        let conversion = OwnedCodecAdvance::begin(source, destination).ok().unwrap();
        let destination = conversion.commit();
        assert_eq!(destination.position(), 2);
        assert_eq!(
            destination
                .history_ranges()
                .iter()
                .map(|(_, rows)| rows)
                .sum::<usize>(),
            2
        );
        assert_eq!(destination_store.occupied_rows(), 2);
    }

    #[test]
    fn recurrent_interior_prefix_remains_unpublished_until_repair() {
        let Some(device) = DeviceCatalog::discover()
            .ok()
            .and_then(|catalog| catalog.open_backend(BackendName::Cpu).ok())
        else {
            return;
        };
        let store = StateStore::new(
            Rc::new(device),
            4,
            4,
            vec![
                ComponentDescriptor::new(LayerRef::Target(0), CodecSpec::dense(DType::F32, 1, 1))
                    .unwrap(),
            ],
            vec![ComponentSpec {
                shape: vec![1],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 1,
                in_flight: 1,
                retained: 0,
            },
        )
        .unwrap();
        let state = store.create().unwrap();
        let advance = OwnedStateAdvance::begin(state, 2).ok().unwrap();
        let OwnedAdvanceResolution::Repair(repair) = advance.commit(1).ok().unwrap() else {
            panic!("interior recurrent prefix must require repair");
        };
        assert_eq!(repair.rows(), 1);
        assert_eq!(store.occupied_rows(), 1);
        let state = repair.abort();
        assert_eq!(state.position(), 0);
        assert_eq!(store.occupied_rows(), 0);
    }
}
