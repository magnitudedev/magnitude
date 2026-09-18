//! Declared storage accounting from the same selected memory plan used by emission.
//! Byte quantities here are allocation sizes, not load/store traffic. Lexical
//! declarations and barrier sites are not native registers or dynamic instances.
use crate::quantity::Count;
use seismic_realization::{dispatch::GroupDispatch, memory::MemoryPlan};
use seismic_realization::{execution::Multiplicity, memory::ControlValue};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Debug)]
pub struct LaunchStorage {
    pub dispatch: GroupDispatch,
    /// Sum of all private array declarations, including disjoint lexical scopes.
    pub declared_private_array_bytes_per_lane: Count,
    pub declared_shared_array_bytes_per_group: Count,
    pub static_barrier_sites: Count,
    /// Collective invocations over the dispatch, not per-lane instructions or
    /// cycles. Unresolved loop/branch facts remain unknown or bounded.
    pub barrier_executions: Count,
    pub native_private_bytes_per_lane: Count,
    pub unmodeled_fragment_operations: Vec<seismic_lang::ir::OperationId>,
}
#[derive(Clone, Debug)]
pub struct StorageAccount {
    pub assumptions: Vec<String>,
    pub launches: Vec<LaunchStorage>,
    /// Current execution retains separately allocated scratch buffers throughout
    /// the invocation; this is the sum of their requested sizes.
    pub retained_scratch_bytes: Count,
    /// Maximum sum of live scratch under serial launch order, inclusive of each
    /// producer and consumer. This describes required lifetimes, not implemented
    /// allocation reuse or an allocator's alignment/padding overhead.
    pub peak_required_scratch_bytes_for_serial_launches: Count,
}
fn count(value: u128) -> Count {
    u64::try_from(value)
        .map(Count::Exact)
        .unwrap_or_else(|_| Count::unknown("storage quantity exceeds u64 range"))
}

pub fn derive(plan: &MemoryPlan, dispatches: &[GroupDispatch]) -> Result<StorageAccount, String> {
    if plan.launches().len() != dispatches.len() {
        return Err("storage accounting requires one dispatch per launch".into());
    }
    let mut multiplicities = HashMap::new();
    let launches = plan
        .launches()
        .iter()
        .zip(dispatches)
        .map(|(launch, dispatch)| {
            if *dispatch
                != GroupDispatch::new(
                    dispatch.work_items,
                    dispatch.lanes_per_item,
                    dispatch.items_per_group,
                )?
            {
                return Err("inconsistent dispatch geometry in storage account".into());
            }
            let mut private = 0u128;
            let mut shared = 0u128;
            for array in &launch.arrays {
                let layout = array.declaration.layout(dispatch)?;
                private += u128::from(layout.private_bytes_per_lane);
                shared += u128::from(layout.shared_bytes_per_group);
            }
            if private != u128::from(launch.declared_private_bytes_per_lane)
                || shared != u128::from(launch.shared_bytes_per_group)
            {
                return Err("storage summary disagrees with selected allocation layouts".into());
            }
            Ok(LaunchStorage {
                dispatch: dispatch.clone(),
                declared_private_array_bytes_per_lane: count(private),
                declared_shared_array_bytes_per_group: count(shared),
                static_barrier_sites: count(launch.barriers.len() as u128),
                barrier_executions: launch
                    .barriers
                    .values()
                    .fold(Count::Exact(0), |sum, barrier| {
                        sum.add(&structured_count(&barrier.executions, &mut multiplicities))
                    })
                    .scale(dispatch.work_items),
                native_private_bytes_per_lane: Count::unknown(
                    "native register allocation, scalar temporaries and spills are unmodeled",
                ),
                unmodeled_fragment_operations: launch.unmodeled_fragments.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    // Sweep lifetime boundaries rather than traversing each allocation for each
    // launch. Consumer accesses happen before release at the next boundary.
    let mut starts = vec![0u128; dispatches.len() + 1];
    let mut ends = vec![0u128; dispatches.len() + 1];
    let mut retained = 0u128;
    for allocation in plan.scratch() {
        let bytes = allocation.bytes as u128;
        starts[allocation.producer] += bytes;
        ends[allocation.consumer + 1] += bytes;
        retained += bytes;
    }
    let mut live = 0u128;
    let mut peak = 0u128;
    for (start, end) in starts.into_iter().zip(ends) {
        live = live
            .checked_sub(end)
            .ok_or("scratch lifetime releases unallocated storage")?
            + start;
        peak = peak.max(live);
    }
    if live != 0 {
        return Err("scratch lifetime extends beyond the invocation".into());
    }
    Ok(StorageAccount {
        assumptions: vec!["runtime validity guards pass and collective sites execute with valid lane participation; convergence qualification is a separate obligation".into()],
        launches,
        retained_scratch_bytes: count(retained),
        peak_required_scratch_bytes_for_serial_launches: count(peak),
    })
}

fn structured_count(
    node: &Arc<Multiplicity<ControlValue>>,
    cache: &mut HashMap<usize, Count>,
) -> Count {
    crate::multiplicity::evaluate(node, cache, &mut |value| match value {
        ControlValue::Integer(s) => s.as_constant(),
        ControlValue::Predicate(e) => match e.kind {
            seismic_lang::ir::ExprKind::Bool(b) => Some(i64::from(b)),
            _ => None,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::{ir::OperationId, types::DType};
    use seismic_realization::{
        dispatch::{TileDeclaration, TilePlacement},
        memory::*,
    };
    use std::collections::BTreeMap;

    fn launch(index: usize) -> LaunchMemory {
        LaunchMemory {
            predecessor: index.checked_sub(1),
            arrays: Vec::new(),
            barriers: BTreeMap::new(),
            declared_private_bytes_per_lane: 0,
            shared_bytes_per_group: 0,
            unmodeled_fragments: Vec::new(),
        }
    }
    fn scratch(index: usize, producer: usize, consumer: usize) -> ScratchAllocation {
        ScratchAllocation {
            index,
            phase: index,
            variable: index,
            dtype: DType::F32,
            elements_per_item: 1,
            work_items: 2,
            parts: 3,
            bytes: 24,
            producer,
            consumer,
        }
    }
    #[test]
    fn allocation_units_are_separate_from_native_storage_and_execution_counts() {
        let mut selected = launch(0);
        for (id, capacity, placement) in [
            (0, 10, TilePlacement::GroupShared),
            (1, 65, TilePlacement::Distributed),
            (2, 4, TilePlacement::Replicated),
        ] {
            selected.arrays.push(ArrayAllocation {
                id: AllocationId {
                    operation: OperationId(id),
                    variable: id,
                    purpose: Purpose::Value,
                },
                declaration: TileDeclaration {
                    symbol: format!("a{id}"),
                    dtype: DType::F32,
                    capacity,
                    placement,
                },
                scope: vec![Scope::Body(OperationId(99))],
            });
        }
        selected.declared_private_bytes_per_lane = 28;
        selected.shared_bytes_per_group = 80;
        selected.barriers.insert(
            BarrierSite {
                operation: OperationId(8),
                variable: 0,
                purpose: BarrierPurpose::Owned,
            },
            Barrier {
                memory: MemorySpace::Threadgroup,
                scope: vec![Scope::Body(OperationId(99))],
                executions: Arc::new(Multiplicity::Unknown {
                    reason: "test runtime trip count".into(),
                }),
            },
        );
        selected.unmodeled_fragments.push(OperationId(9));
        let plan = MemoryPlan::new(vec![selected], vec![]).unwrap();
        let account = derive(&plan, &[GroupDispatch::new(5, 32, 2).unwrap()]).unwrap();
        let launch = &account.launches[0];
        assert_eq!(
            launch.declared_private_array_bytes_per_lane,
            Count::Exact(28)
        );
        assert_eq!(
            launch.declared_shared_array_bytes_per_group,
            Count::Exact(80)
        );
        assert_eq!(launch.static_barrier_sites, Count::Exact(1));
        assert!(launch.barrier_executions.bounds().is_none());
        assert!(launch.native_private_bytes_per_lane.bounds().is_none());
        assert_eq!(launch.unmodeled_fragment_operations, [OperationId(9)]);
        let empty = derive(&plan, &[GroupDispatch::new(0, 32, 2).unwrap()]).unwrap();
        assert_eq!(empty.launches[0].barrier_executions, Count::Exact(0));
        assert!(derive(&plan, &[GroupDispatch::new(5, 32, 4).unwrap()]).is_err());
    }
    #[test]
    fn scratch_retention_is_not_confused_with_required_lifetime_overlap() {
        let dispatches = vec![GroupDispatch::new(2, 32, 1).unwrap(); 4];
        for (allocations, expected_peak) in [
            (vec![scratch(0, 0, 1), scratch(1, 2, 3)], 24),
            (vec![scratch(0, 0, 3), scratch(1, 1, 2)], 48),
        ] {
            let plan = MemoryPlan::new((0..4).map(launch).collect(), allocations).unwrap();
            let account = derive(&plan, &dispatches).unwrap();
            assert_eq!(account.retained_scratch_bytes, Count::Exact(48));
            assert_eq!(
                account.peak_required_scratch_bytes_for_serial_launches,
                Count::Exact(expected_peak)
            );
        }
    }
    #[test]
    fn malformed_memory_contracts_are_rejected() {
        let mut incorrect_size = scratch(0, 0, 1);
        incorrect_size.bytes = 20;
        assert!(MemoryPlan::new(vec![launch(0), launch(1)], vec![incorrect_size]).is_err());
        let mut unordered = launch(1);
        unordered.predecessor = None;
        assert!(MemoryPlan::new(vec![launch(0), unordered], vec![scratch(0, 0, 1)]).is_err());
        let mut cycle = launch(0);
        cycle.predecessor = Some(0);
        assert!(MemoryPlan::new(vec![cycle], vec![]).is_err());
        let mut incorrect_summary = launch(0);
        incorrect_summary.shared_bytes_per_group = 4;
        let plan = MemoryPlan::new(vec![incorrect_summary], vec![]).unwrap();
        assert!(derive(&plan, &[GroupDispatch::new(1, 32, 1).unwrap()]).is_err());
        assert!(derive(&plan, &[]).is_err());
    }
}
