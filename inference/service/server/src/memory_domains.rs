//! The public names of memory domains: the topology's stable store
//! identities. Host RAM is the `system` domain; every other store is named
//! by its identity, whichever backend reaches it, so the views of one
//! physical device's memory (a GPU seen through CUDA and Vulkan) share one
//! name. Assessments, hardware snapshots, merged worker observations and
//! instance allocations share these names.

use magnitude_engine::census::{AllocationCensus, BankResize, ExpertResidency, ShrinkCause};
use magnitude_service_contracts::MemoryDomainId;
use magnitude_service_contracts::models::{
    ModelInstanceAllocation, ModelInstanceMemoryDomain, ModelStreamingStatus, ModelStreamingStatusParts, OutOfBounds,
    PromptGroupPlacesParts, StreamingHostStatusParts, StreamingResize, StreamingShrinkCause, StreamingTableStatusParts,
    StreamingTierStatus,
};
use seismic::{DeviceTopology, StoreId, StoreIdentity};
use std::num::{NonZeroU64, NonZeroUsize};

/// The name of the store `identity` names, in any process.
pub fn store_identity_id(identity: &StoreIdentity) -> MemoryDomainId {
    match identity {
        StoreIdentity::HostRam => MemoryDomainId::system(),
        identity => MemoryDomainId::new(identity.to_string()),
    }
}

/// The name of `topology`'s store `store`.
pub fn store_domain_id(topology: &DeviceTopology, store: StoreId) -> MemoryDomainId {
    store_identity_id(
        &topology
            .store_identity(store)
            .expect("store belongs to the topology"),
    )
}

/// A census count of places as the contract states it.
fn places(places: NonZeroUsize) -> NonZeroU64 {
    NonZeroU64::try_from(places).expect("a usize count fits in u64")
}

/// An instance's allocation from its engine's census; a census outside the contract's bounds
/// is refused.
pub fn instance_allocation(
    context_window_tokens: u32,
    census: &AllocationCensus,
) -> Result<ModelInstanceAllocation, OutOfBounds> {
    let serving = &census.serving;
    Ok(ModelInstanceAllocation {
        context_window_tokens,
        memory_domains: census
            .stores
            .iter()
            .map(|store| ModelInstanceMemoryDomain {
                memory_domain_id: store_identity_id(&store.store),
                model_bytes: store.model_bytes,
                context_bytes: store.context_bytes,
                compute_bytes: store.compute_bytes,
                auxiliary_bytes: store.auxiliary_bytes,
            })
            .collect(),
        prompt_group_places: PromptGroupPlacesParts {
            held: places(serving.group_places),
            capacity: places(serving.group_capacity),
        }
        .try_into()?,
        streaming: serving.experts.as_ref().map(streaming_status).transpose()?,
    })
}

/// Where an elastic bank's experts are, from the engine's census of them, each holding with
/// its last resize.
fn streaming_status(residency: &ExpertResidency) -> Result<ModelStreamingStatus, OutOfBounds> {
    ModelStreamingStatusParts {
        experts: residency.experts,
        slots: residency.slots,
        floor_slots: residency.floor,
        ceiling_slots: residency.ceiling,
        tiers: residency
            .stores
            .iter()
            .map(|(store, bytes)| StreamingTierStatus {
                memory_domain_id: store_identity_id(store),
                bytes: *bytes,
            })
            .collect(),
        host: residency
            .host
            .as_ref()
            .map(|host| {
                StreamingHostStatusParts {
                    memory_domain_id: store_identity_id(&StoreIdentity::HostRam),
                    slots: host.slots,
                    ceiling_slots: host.ceiling,
                    bytes: host.bytes,
                    last_resize: host.last_resize.as_ref().map(streaming_resize),
                }
                .try_into()
            })
            .transpose()?,
        disk_bytes: residency.disk_bytes,
        tables: residency
            .tables
            .iter()
            .map(|table| {
                StreamingTableStatusParts {
                    records: table.records,
                    floor_records: table.floor,
                    ceiling_records: table.ceiling,
                    bytes: table.bytes,
                    last_resize: table.last_resize.as_ref().map(streaming_resize),
                }
                .try_into()
            })
            .collect::<Result<_, _>>()?,
        last_resize: residency.last_resize.as_ref().map(streaming_resize),
    }
    .try_into()
}

/// A holding's finished resize on the contract, as the engine's census states it.
fn streaming_resize(resize: &BankResize) -> StreamingResize {
    match *resize {
        BankResize::Shrank { cause, to, by, at_unix_ms } => StreamingResize::Shrank {
            cause: match cause {
                ShrinkCause::Reclaim => StreamingShrinkCause::Reclaim,
                ShrinkCause::LiveClaim => StreamingShrinkCause::LiveClaim,
                ShrinkCause::Rebalance => StreamingShrinkCause::Rebalance,
                ShrinkCause::Measurement => StreamingShrinkCause::Measurement,
            },
            to,
            by,
            at_unix_ms,
        },
        BankResize::Grew { from, by, at_unix_ms } => StreamingResize::Grew { from, by, at_unix_ms },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_engine::census::{ExpertResidency, HostResidency, ServingCensus, TableResidency};

    /// The allocation states what the engine's census says of how it serves:
    /// its prompt groups' places, and its bank's slots, stores by name, disk
    /// bytes and last resize with its cause.
    #[test]
    fn an_allocation_states_the_engines_serving_census() {
        let device = seismic::StoreIdentity::DeviceMemory(seismic::PhysicalIdentity::Uuid([3; 16]));
        let positive = |value| NonZeroU64::new(value).unwrap();
        let count = |value| NonZeroUsize::new(value).unwrap();
        let census = AllocationCensus {
            stores: Vec::new(),
            serving: ServingCensus {
                group_places: count(3),
                group_capacity: count(8),
                experts: Some(ExpertResidency {
                    experts: positive(256),
                    slots: 64,
                    floor: 26,
                    ceiling: 200,
                    stores: vec![(device.clone(), 700), (seismic::StoreIdentity::HostRam, 300)],
                    host: Some(HostResidency {
                        slots: 20,
                        ceiling: 256,
                        bytes: 650,
                        fill: magnitude_engine::census::TierFill::Filled,
                        last_resize: Some(BankResize::Grew { from: 19, by: positive(1), at_unix_ms: 11 }),
                    }),
                    disk_bytes: 900,
                    tables: vec![TableResidency {
                        records: 40,
                        floor: 1,
                        ceiling: positive(42),
                        bytes: 500,
                        last_resize: Some(BankResize::Shrank {
                            cause: ShrinkCause::Reclaim,
                            to: 40,
                            by: positive(1),
                            at_unix_ms: 12,
                        }),
                    }],
                    last_resize: Some(BankResize::Shrank {
                        cause: ShrinkCause::Measurement,
                        to: 64,
                        by: positive(8),
                        at_unix_ms: 10,
                    }),
                }),
            },
        };
        let allocation = instance_allocation(4_096, &census).unwrap();
        assert_eq!(
            serde_json::to_value(&allocation).unwrap(),
            serde_json::json!({
                "contextWindowTokens": 4096,
                "memoryDomains": [],
                "promptGroupPlaces": { "held": 3, "capacity": 8 },
                "streaming": {
                    "experts": 256,
                    "slots": 64,
                    "floorSlots": 26,
                    "ceilingSlots": 200,
                    "tiers": [
                        { "memoryDomainId": store_identity_id(&device), "bytes": 700 },
                        { "memoryDomainId": "system", "bytes": 300 },
                    ],
                    "host": {
                        "memoryDomainId": "system",
                        "slots": 20, "ceilingSlots": 256, "bytes": 650,
                        "lastResize": { "_tag": "Grew", "from": 19, "by": 1, "atUnixMs": 11 },
                    },
                    "diskBytes": 900,
                    "tables": [{
                        "records": 40, "floorRecords": 1, "ceilingRecords": 42, "bytes": 500,
                        "lastResize": { "_tag": "Shrank", "cause": "reclaim", "to": 40, "by": 1, "atUnixMs": 12 },
                    }],
                    "lastResize": { "_tag": "Shrank", "cause": "measurement", "to": 64, "by": 8, "atUnixMs": 10 },
                },
            })
        );
        // A census outside the contract's bounds is refused: a bound out of order.
        let refused = |change: &dyn Fn(&mut AllocationCensus)| {
            let mut changed = census.clone();
            change(&mut changed);
            instance_allocation(4_096, &changed).unwrap_err()
        };
        fn experts(census: &mut AllocationCensus) -> &mut ExpertResidency {
            census.serving.experts.as_mut().unwrap()
        }
        assert_eq!(
            refused(&|census| census.serving.group_places = count(9)).0,
            "a prompt group holds at most its capacity"
        );
        assert_eq!(
            refused(&|census| experts(census).slots = 201).0,
            "an expert bank holds between its floor and its ceiling, at most every expert"
        );
    }
}
