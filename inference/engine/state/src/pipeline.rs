//! Ordered local transactions sharing one logical sequence extent.
//! This composes state ownership, not devices, physical execution or placement.
use crate::advance::PreparedStateCommit;
use crate::{Error, OwnedAdvanceResolution, OwnedStateAdvance, SequenceState};
use std::rc::Rc;

/// Refusals return unchanged accepted owners; boxing is failure-only.
pub type PipelineStateRefusal<T> = Box<(T, Error)>;

/// Distinct local stores in caller-supplied order, with equal logical extents.
/// This does not attest model, placement or execution-device identity.
pub struct PipelineSequenceState {
    stages: Vec<SequenceState>,
}

fn compatible(stages: &[SequenceState]) -> bool {
    let Some(first) = stages.first() else {
        return false;
    };
    stages.iter().enumerate().all(|(i, state)| {
        state.position() == first.position()
            && state.expected_end() == first.expected_end()
            && stages[..i]
                .iter()
                .all(|prior| !Rc::ptr_eq(&prior.store, &state.store))
    })
}

impl PipelineSequenceState {
    pub fn new(
        stages: Vec<SequenceState>,
    ) -> Result<Self, PipelineStateRefusal<Vec<SequenceState>>> {
        if !compatible(&stages) {
            return Err(Box::new((
                stages,
                Error::from("pipeline sources require distinct stores and equal logical extents"),
            )));
        }
        Ok(Self { stages })
    }
    pub fn stages(&self) -> &[SequenceState] {
        &self.stages
    }
    pub fn into_stages(self) -> Vec<SequenceState> {
        self.stages
    }
    pub fn position(&self) -> usize {
        self.stages[0].position()
    }

    /// Refusal at any store unwinds earlier reservations in original stage order.
    pub fn begin(self, rows: usize) -> Result<PipelineStateAdvance, PipelineStateRefusal<Self>> {
        let mut remaining = self.stages.into_iter();
        let mut advances = Vec::with_capacity(remaining.len());
        while let Some(state) = remaining.next() {
            match OwnedStateAdvance::begin(state, rows) {
                Ok(advance) => advances.push(advance),
                Err((state, error)) => {
                    let mut stages: Vec<_> =
                        advances.into_iter().map(OwnedStateAdvance::abort).collect();
                    stages.push(state);
                    stages.extend(remaining);
                    return Err(Box::new((Self { stages }, error)));
                }
            }
        }
        Ok(PipelineStateAdvance { stages: advances })
    }
}

/// Owned ordinary advances, retained together until logical reconciliation.
/// The caller must observe all physical completions before committing the group.
pub struct PipelineStateAdvance {
    stages: Vec<OwnedStateAdvance>,
}

impl PipelineStateAdvance {
    /// Rejoin checked ordinary advances; speculative/local extents cannot disagree.
    pub fn new(
        stages: Vec<OwnedStateAdvance>,
    ) -> Result<Self, PipelineStateRefusal<Vec<OwnedStateAdvance>>> {
        let valid = stages.first().is_some_and(|first| {
            stages.iter().enumerate().all(|(i, stage)| {
                stage.position() == first.position()
                    && stage.rows() == first.rows()
                    && stage.bindings().stop == stage.rows()
                    && stage.source().expected_end() == first.source().expected_end()
                    && stages[..i]
                        .iter()
                        .all(|prior| !Rc::ptr_eq(&prior.source().store, &stage.source().store))
            })
        });
        if !valid {
            return Err(Box::new((
                stages,
                Error::from(
                    "pipeline advances require distinct stores and equal non-speculative extents",
                ),
            )));
        }
        Ok(Self { stages })
    }
    pub fn stages(&self) -> &[OwnedStateAdvance] {
        &self.stages
    }
    pub fn into_stages(self) -> Vec<OwnedStateAdvance> {
        self.stages
    }
    pub fn abort(self) -> PipelineSequenceState {
        PipelineSequenceState {
            stages: self
                .stages
                .into_iter()
                .map(OwnedStateAdvance::abort)
                .collect(),
        }
    }

    /// Call only after every physical stage completed. Preflight ALL commits
    /// before publishing ANY; a later refusal returns every original source.
    /// No callback or recoverable failure occurs between publications.
    /// This does not roll back completed GPU writes or make a failed request reusable.
    pub fn commit(
        self,
        accepted: usize,
    ) -> Result<PipelineSequenceState, PipelineStateRefusal<PipelineSequenceState>> {
        let mut remaining = self.stages.into_iter();
        let mut prepared: Vec<PreparedStateCommit> = Vec::with_capacity(remaining.len());
        while let Some(advance) = remaining.next() {
            match advance.prepare_commit(accepted) {
                Ok(commit) => prepared.push(commit),
                Err(refusal) => {
                    let (state, error) = *refusal;
                    let mut stages: Vec<_> = prepared
                        .into_iter()
                        .map(PreparedStateCommit::abort)
                        .collect();
                    stages.push(state);
                    stages.extend(remaining.map(OwnedStateAdvance::abort));
                    return Err(Box::new((PipelineSequenceState { stages }, error)));
                }
            }
        }
        let stages = prepared
            .into_iter()
            .map(|commit| match commit.publish() {
                OwnedAdvanceResolution::Committed(state)
                | OwnedAdvanceResolution::Aborted(state) => state,
            })
            .collect();
        Ok(PipelineSequenceState { stages })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BankCapacity, ComponentSpec, StateStore};
    use seismic::{BackendName, DType, Device, DeviceCatalog};

    fn stores(contexts: &[usize], recurrent: &[bool]) -> Vec<Rc<StateStore>> {
        let device = Rc::new(
            DeviceCatalog::discover()
                .unwrap()
                .open_backend(BackendName::Cpu)
                .unwrap(),
        );
        contexts
            .iter()
            .zip(recurrent)
            .map(|(&context, &recurrent)| store(device.clone(), context, recurrent))
            .collect()
    }
    fn store(device: Rc<Device>, context: usize, recurrent: bool) -> Rc<StateStore> {
        StateStore::new(
            device,
            context,
            context,
            vec![],
            if recurrent {
                vec![ComponentSpec {
                    shape: vec![256],
                    dtype: DType::F32,
                }]
            } else {
                vec![]
            },
            BankCapacity {
                active: 2,
                in_flight: 2,
                retained: 0,
            },
        )
        .unwrap()
        .0
    }
    fn sources(stores: &[Rc<StateStore>]) -> PipelineSequenceState {
        PipelineSequenceState::new(stores.iter().map(|s| s.create().unwrap()).collect())
            .unwrap_or_else(|e| panic!("{}", e.1))
    }
    fn begin(state: PipelineSequenceState, rows: usize) -> PipelineStateAdvance {
        state.begin(rows).unwrap_or_else(|e| panic!("{}", e.1))
    }
    fn positions(state: &PipelineSequenceState) -> Vec<usize> {
        state.stages().iter().map(SequenceState::position).collect()
    }
    #[test]
    fn ordered_state_groups_support_one_two_three_four_and_eight_stores() {
        // CPU-only ownership tests, NOT N-GPU execution qualification.
        for n in [1, 2, 3, 4, 8] {
            let stores = stores(&vec![8; n], &vec![true; n]);
            let mut state = sources(&stores);
            for position in 1..=2 {
                let advance = begin(state, 1);
                for (advance, store) in advance.stages().iter().zip(&stores) {
                    assert!(advance.belongs_to(store));
                }
                state = advance.commit(1).unwrap_or_else(|e| panic!("{}", e.1));
                assert_eq!(positions(&state), vec![position; n]);
            }
        }
    }
    #[test]
    fn refusal_at_each_of_three_reservations_recovers_all_sources() {
        for failed in 0..3 {
            let mut contexts = [8; 3];
            contexts[failed] = 1;
            let stores = stores(&contexts, &[true; 3]);
            let (state, _) = *sources(&stores)
                .begin(2)
                .err()
                .expect("short context refuses");
            assert_eq!(positions(&state), vec![0; 3]);
            assert_eq!(positions(&begin(state, 1).abort()), vec![0; 3]);
        }
    }
    #[test]
    fn refusal_at_any_commit_preflight_publishes_no_stage() {
        // Earlier stateless stages accept an interior prefix; only the selected
        // recurrent stage refuses it. Include first, middle and LAST refusal.
        for refused in 0..3 {
            let mut recurrent = [false; 3];
            recurrent[refused] = true;
            let stores = stores(&[8; 3], &recurrent);
            let charged: Vec<_> = stores.iter().map(|s| s.committed_bytes()).collect();
            let (state, _) = *begin(sources(&stores), 2)
                .commit(1)
                .err()
                .expect("recurrent stage refuses interior prefix");
            assert_eq!(positions(&state), vec![0; 3]);
            for (source, store) in state.stages().iter().zip(&stores) {
                assert!(source.belongs_to(store));
            }
            assert_eq!(
                stores
                    .iter()
                    .map(|s| s.committed_bytes())
                    .collect::<Vec<_>>(),
                charged
            );
            let committed = begin(state, 2)
                .commit(2)
                .unwrap_or_else(|e| panic!("{}", e.1));
            assert_eq!(positions(&committed), vec![2; 3]);
        }
    }
    #[test]
    fn rejoining_owned_advances_preserves_stage_order_and_publication() {
        for n in [2, 3, 4, 8] {
            let stores = stores(&vec![8; n], &vec![true; n]);
            let stages = begin(sources(&stores), 2).into_stages();
            let joined = PipelineStateAdvance::new(stages).unwrap_or_else(|e| panic!("{}", e.1));
            for (advance, store) in joined.stages().iter().zip(&stores) {
                assert!(advance.belongs_to(store));
            }
            let state = joined.commit(2).unwrap_or_else(|e| panic!("{}", e.1));
            assert_eq!(positions(&state), vec![2; n]);
            assert_eq!(
                positions(
                    &begin(state, 1)
                        .commit(1)
                        .unwrap_or_else(|e| panic!("{}", e.1))
                ),
                vec![3; n]
            );
        }
    }
    #[test]
    fn single_store_ordinary_acceptance_is_unchanged() {
        for recurrent in [false, true] {
            for accepted in 0..=3 {
                let stores = stores(&[8, 8], &[recurrent; 2]);
                let ordinary = OwnedStateAdvance::begin(stores[0].create().unwrap(), 2)
                    .unwrap_or_else(|e| panic!("{}", e.1))
                    .commit(accepted);
                let group = begin(sources(&stores[1..]), 2).commit(accepted);
                if accepted > 2 || (recurrent && accepted == 1) {
                    let (ordinary, error) = ordinary.err().expect("ordinary refusal");
                    let (group, grouped_error) = *group.err().expect("group refusal");
                    assert_eq!(error, grouped_error);
                    assert_eq!(ordinary.position(), 0);
                    assert_eq!(positions(&group), [0]);
                } else {
                    let ordinary = match ordinary.unwrap_or_else(|e| panic!("{}", e.1)) {
                        OwnedAdvanceResolution::Committed(s)
                        | OwnedAdvanceResolution::Aborted(s) => s,
                    };
                    let group = group.unwrap_or_else(|e| panic!("{}", e.1));
                    assert_eq!(ordinary.position(), accepted);
                    assert_eq!(positions(&group), [accepted]);
                    assert_eq!(ordinary.expected_end(), group.stages()[0].expected_end());
                    assert_eq!(ordinary.tape, group.stages()[0].tape);
                }
            }
        }
    }
    #[test]
    fn zero_and_invalid_acceptance_restore_every_source() {
        let stores = stores(&[8; 3], &[true; 3]);
        let (state, _) = *begin(sources(&stores), 1).commit(2).err().unwrap();
        assert_eq!(positions(&state), vec![0; 3]);
        let state = begin(state, 1)
            .commit(0)
            .unwrap_or_else(|e| panic!("{}", e.1));
        assert_eq!(positions(&state), vec![0; 3]);
    }
    #[test]
    fn empty_duplicate_store_and_mismatched_extents_are_refused() {
        assert!(PipelineSequenceState::new(vec![]).is_err());
        let stores = stores(&[8; 3], &[false; 3]);
        assert!(PipelineSequenceState::new(vec![
            stores[0].create().unwrap(),
            stores[1].create().unwrap(),
            stores[0].create().unwrap()
        ])
        .is_err());
        let mut states = sources(&stores).into_stages();
        states[2].anticipate(2).unwrap();
        assert!(PipelineSequenceState::new(states).is_err());
        assert!(PipelineStateAdvance::new(vec![]).is_err());
    }
    #[test]
    fn joining_mismatched_or_speculative_advances_is_refused() {
        let stores = stores(&[8; 3], &[true; 3]);
        let a = OwnedStateAdvance::begin(stores[0].create().unwrap(), 1)
            .unwrap_or_else(|e| panic!("{}", e.1));
        let b = OwnedStateAdvance::begin(stores[1].create().unwrap(), 2)
            .unwrap_or_else(|e| panic!("{}", e.1));
        assert!(PipelineStateAdvance::new(vec![a, b]).is_err());
        let a = OwnedStateAdvance::begin_speculative(stores[0].create().unwrap(), 2, 1)
            .unwrap_or_else(|e| panic!("{}", e.1));
        assert!(PipelineStateAdvance::new(vec![a]).is_err());
    }
    #[test]
    fn joining_foreign_logical_extents_or_duplicate_store_advances_is_refused() {
        let stores = stores(&[8; 2], &[true; 2]);
        let make = |state| OwnedStateAdvance::begin(state, 1).unwrap_or_else(|e| panic!("{}", e.1));
        let duplicate = vec![
            make(stores[0].create().unwrap()),
            make(stores[0].create().unwrap()),
        ];
        assert!(PipelineStateAdvance::new(duplicate).is_err());
        let advanced = match make(stores[1].create().unwrap())
            .commit(1)
            .unwrap_or_else(|e| panic!("{}", e.1))
        {
            OwnedAdvanceResolution::Committed(s) | OwnedAdvanceResolution::Aborted(s) => s,
        };
        let a = make(stores[0].create().unwrap());
        let b = make(advanced);
        let (advances, _) = *PipelineStateAdvance::new(vec![a, b])
            .err()
            .expect("different positions refuse");
        assert_eq!(
            advances
                .iter()
                .map(OwnedStateAdvance::position)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        drop(advances);
        let mut anticipated = stores[1].create().unwrap();
        anticipated.anticipate(2).unwrap();
        assert!(PipelineStateAdvance::new(vec![
            make(stores[0].create().unwrap()),
            make(anticipated)
        ])
        .is_err());
    }
    #[test]
    fn repeated_abort_and_drop_return_successor_banks_without_charge_growth() {
        let stores = stores(&[8; 3], &[true; 3]);
        let charged: Vec<_> = stores.iter().map(|s| s.committed_bytes()).collect();
        let mut state = sources(&stores);
        for _ in 0..100 {
            state = begin(state, 1).abort();
            assert_eq!(
                stores
                    .iter()
                    .map(|s| s.committed_bytes())
                    .collect::<Vec<_>>(),
                charged
            );
        }
        drop(begin(state, 1));
        let advance = begin(sources(&stores), 1);
        for stage in advance.stages() {
            assert_eq!(stage.bindings().following_bank, 1);
        }
    }
}
