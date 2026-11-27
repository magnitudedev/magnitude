//! Mandatory Deterministic Acceptance Replay Suite for `seismic-io` v2
//! (Section 7 of the Astra-reviewed architecture specification):
//!
//! 1. **Census Invariance (`1×20` vs `20×1`)**: work-conserving batch coalescing
//!    produces identical logical completion counts and zero double-counting.
//! 2. **Joined-Waiter Cancellation**: Subscriber A and B join one `ContentKey`;
//!    A cancels while `Issued` -> physical op completes, destination `BufferLease`
//!    stays pinned until B retires.
//! 3. **Mid-Batch Partial Issue Failure**: driver fails after copy 1 of 4 ->
//!    `Submitting -> Quarantined`, zero destination slots returned early, backing
//!    retained until explicit teardown.
//! 4. **Inverted Producer Completions**: independent `(DeviceId, ProducerId)`
//!    streams B and A complete out of order -> never corrupt or erase each
//!    other's fences.
//! 5. **Reader-Retirement Overlap**: slot overwrite blocked while a reserved
//!    unpublished pass still holds a reader pin.
//! 6. **1:1 Physical Debit/Credit Conservation**: randomized submit / reprice /
//!    cancel / transient-rollback / refuse / quarantine / retire fuzzing returns
//!    `in_flight_bytes == 0` and `CoupledSet` bus credits == 0 on quiescence,
//!    plus hot-path latency check.
//! 7. **Residual Speculative Service-Time Bound, Coupled-Bus Preemption,
//!    Fence-Bound `BusReservation`, and Multi-Hop Priority Inheritance**.

use seismic_io::{
    BatchCompatibility, BufferLease, BusCoupling, BusDirection, ContentKey, DeviceId, DomainId,
    Fabric, FabricSpec, FenceCapability, FenceError, GlobalUrgency, IntermediateSlotReserve,
    IoPlane, LeasePool, LinkCurve, LinkId, LinkKind, LinkSpec, MultiProducerTimeline, NumaNodeId,
    PipelineHandle, PlacementRequirement, PriorityClass, ProducerId, ReaderRetirementTracker,
    RepresentationId, RootComplexId, ScheduleKey, SharedBusId, SharedBusSpec, SlotKey,
    TimelineObserver, TimelinePoint, TransferOutcome, TransferRequest, VolumeId,
};
use std::collections::HashSet;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct TrackingPool {
    released: Mutex<HashSet<(u32, u64)>>,
    quarantined: Mutex<HashSet<(u32, u64)>>,
}

impl TrackingPool {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            released: Mutex::new(HashSet::new()),
            quarantined: Mutex::new(HashSet::new()),
        })
    }

    fn is_released(&self, slot: u32, gen: u64) -> bool {
        self.released.lock().unwrap().contains(&(slot, gen))
    }

    fn is_quarantined(&self, slot: u32, gen: u64) -> bool {
        self.quarantined.lock().unwrap().contains(&(slot, gen))
    }

    fn released_count(&self) -> usize {
        self.released.lock().unwrap().len()
    }
}

impl LeasePool for TrackingPool {
    fn release_slot(&self, slot: u32, generation: u64) {
        let inserted = self.released.lock().unwrap().insert((slot, generation));
        assert!(inserted, "double-release of ({slot}, {generation})");
    }

    fn quarantine_slot(&self, slot: u32, generation: u64) {
        self.quarantined.lock().unwrap().insert((slot, generation));
    }
}

fn discrete_fabric() -> Fabric {
    let ddr0 = SharedBusId::HostDdr(NumaNodeId(0));
    let pcie0 = SharedBusId::PcieRoot(RootComplexId(0));
    Fabric::build(FabricSpec {
        revision: 1,
        buses: vec![
            SharedBusSpec {
                id: ddr0,
                peak_bps: 60e9,
                uncertainty_rel: 0.03,
            },
            SharedBusSpec {
                id: pcie0,
                peak_bps: 32e9,
                uncertainty_rel: 0.03,
            },
        ],
        links: vec![
            LinkSpec {
                id: LinkId(0),
                kind: LinkKind::NvmeDirect,
                src: DomainId::Storage(VolumeId(0)),
                dst: DomainId::HostRam(NumaNodeId(0)),
                curve: LinkCurve {
                    latency_s: 200e-6,
                    peak_bps: 6e9,
                    knee_bytes: 4 * 1024 * 1024,
                    due_depth_bytes: 32 * 1024 * 1024,
                    max_in_flight_ops: 16,
                    alignment_bytes: 4096,
                    uncertainty_rel: 0.05,
                },
                couplings: vec![
                    BusCoupling {
                        bus: ddr0,
                        direction: BusDirection::Write,
                        weight: 1.0,
                    },
                    BusCoupling {
                        bus: pcie0,
                        direction: BusDirection::FullDuplexD2H,
                        weight: 1.0,
                    },
                ],
            },
            LinkSpec {
                id: LinkId(1),
                kind: LinkKind::CopyEngineH2D,
                src: DomainId::HostRam(NumaNodeId(0)),
                dst: DomainId::DeviceVram(DeviceId(0)),
                curve: LinkCurve {
                    latency_s: 5.1e-6,
                    peak_bps: 25e9,
                    knee_bytes: 8 * 1024 * 1024,
                    due_depth_bytes: 64 * 1024 * 1024,
                    max_in_flight_ops: 32,
                    alignment_bytes: 256,
                    uncertainty_rel: 0.03,
                },
                couplings: vec![
                    BusCoupling {
                        bus: ddr0,
                        direction: BusDirection::Read,
                        weight: 1.0,
                    },
                    BusCoupling {
                        bus: pcie0,
                        direction: BusDirection::FullDuplexH2D,
                        weight: 1.0,
                    },
                ],
            },
        ],
    })
    .expect("valid discrete fabric")
}

fn make_lease(domain: DomainId, slot: u32, gen: u64, len: usize, pool: &Arc<TrackingPool>) -> BufferLease {
    BufferLease::from_pool(
        domain,
        NonNull::dangling(),
        1,
        (slot as u64) * (len as u64),
        len,
        4096,
        slot,
        gen,
        Arc::clone(pool) as Arc<dyn LeasePool>,
    )
}

/// Scenario 1: Census Invariance (`1×20` vs `20×1` work-conserving coalescing).
/// Identical 20-expert demand arriving either all-at-once (1 coalesced wave)
/// or staggered (1 initial wave + 19 coalesced while wave 1 is in flight)
/// produces the exact same completed ops, completed bytes, and zero double-counting.
#[test]
fn scenario_1_census_invariance_1x20_vs_20x1() {
    let vram = DomainId::DeviceVram(DeviceId(0));
    let expert_bytes = 2 * 1024 * 1024u64;
    let compat = BatchCompatibility::SharedConsumer { step: 12, layer: 4 };

    // Case A: 20 arrivals ready simultaneously -> 1 coalesced HW batch.
    let plane_a = IoPlane::new(discrete_fabric());
    let pool_a = TrackingPool::new();
    let mut handles_a = Vec::new();
    for i in 0..20u32 {
        let h = plane_a
            .submit(TransferRequest {
                link: LinkId(1),
                key: ScheduleKey::blocking(GlobalUrgency(0), 12, 4, i as u64),
                bytes: expert_bytes,
                content_key: Some(ContentKey {
                    asset_id: 1,
                    slice_id: i as u64,
                    representation_id: RepresentationId(0),
                    source_gen: 1,
                }),
                placement: PlacementRequirement::AliasInDomain(vram),
                compatibility: compat,
                src_lease: None,
                dst_lease: make_lease(vram, i, 1, expert_bytes as usize, &pool_a),
                direct: true,
            })
            .unwrap();
        handles_a.push(h);
    }
    let batch_a = plane_a
        .claim_next_batch(LinkId(1), 64 * 1024 * 1024, 1.0)
        .expect("coalesced 1x20 batch");
    assert_eq!(batch_a.ops.len(), 20, "all 20 compatible Blocking ops coalesce into 1 batch");
    plane_a.confirm_issued(batch_a.batch_id, None);
    plane_a.retire_issued(batch_a.batch_id);
    for h in handles_a {
        assert!(matches!(h.outcome(), Some(TransferOutcome::Completed { .. })));
    }
    let ledger_a = plane_a.link_ledger(LinkId(1)).unwrap();

    // Case B: First expert arrives -> issued immediately (zero artificial wait!).
    // While wave 1 runs on the copy engine, remaining 19 experts arrive ->
    // coalesced into a single second wave.
    let plane_b = IoPlane::new(discrete_fabric());
    let pool_b = TrackingPool::new();
    let mut handles_b = Vec::new();
    handles_b.push(
        plane_b
            .submit(TransferRequest {
                link: LinkId(1),
                key: ScheduleKey::blocking(GlobalUrgency(0), 12, 4, 0),
                bytes: expert_bytes,
                content_key: Some(ContentKey {
                    asset_id: 1,
                    slice_id: 0,
                    representation_id: RepresentationId(0),
                    source_gen: 1,
                }),
                placement: PlacementRequirement::AliasInDomain(vram),
                compatibility: compat,
                src_lease: None,
                dst_lease: make_lease(vram, 0, 1, expert_bytes as usize, &pool_b),
                direct: true,
            })
            .unwrap(),
    );
    let wave1 = plane_b
        .claim_next_batch(LinkId(1), 64 * 1024 * 1024, 1.0)
        .expect("immediate wave 1 with zero artificial wait");
    assert_eq!(wave1.ops.len(), 1);
    plane_b.confirm_issued(wave1.batch_id, None);

    for i in 1..20u32 {
        handles_b.push(
            plane_b
                .submit(TransferRequest {
                    link: LinkId(1),
                    key: ScheduleKey::blocking(GlobalUrgency(0), 12, 4, i as u64),
                    bytes: expert_bytes,
                    content_key: Some(ContentKey {
                        asset_id: 1,
                        slice_id: i as u64,
                        representation_id: RepresentationId(0),
                        source_gen: 1,
                    }),
                    placement: PlacementRequirement::AliasInDomain(vram),
                    compatibility: compat,
                    src_lease: None,
                    dst_lease: make_lease(vram, i, 1, expert_bytes as usize, &pool_b),
                    direct: true,
                })
                .unwrap(),
        );
    }
    let wave2 = plane_b
        .claim_next_batch(LinkId(1), 64 * 1024 * 1024, 1.0)
        .expect("wave 2 coalesces all 19 accumulated arrivals");
    assert_eq!(wave2.ops.len(), 19);
    plane_b.confirm_issued(wave2.batch_id, None);
    plane_b.retire_issued(wave1.batch_id);
    plane_b.retire_issued(wave2.batch_id);
    for h in handles_b {
        assert!(matches!(h.outcome(), Some(TransferOutcome::Completed { .. })));
    }
    let ledger_b = plane_b.link_ledger(LinkId(1)).unwrap();

    // Exact invariance: completed ops & bytes match identically with zero double-counting.
    assert_eq!(ledger_a.completed_ops_by_class, ledger_b.completed_ops_by_class);
    assert_eq!(ledger_a.completed_bytes_by_class, ledger_b.completed_bytes_by_class);
    assert_eq!(ledger_a.completed_ops_by_class[PriorityClass::Blocking.index()], 20);
    assert_eq!(ledger_a.total_in_flight_bytes(), 0);
    assert_eq!(ledger_b.total_in_flight_bytes(), 0);
    assert_eq!(pool_a.released_count(), 20);
    assert_eq!(pool_b.released_count(), 20);
}

/// Scenario 2: Joined-Waiter Cancellation.
/// Subscriber A and B join one `ContentKey`; A cancels while `Issued` ->
/// physical op completes, destination `BufferLease` stays pinned until B retires.
#[test]
fn scenario_2_joined_waiter_cancellation() {
    let plane = IoPlane::new(discrete_fabric());
    let pool = TrackingPool::new();
    let vram = DomainId::DeviceVram(DeviceId(0));
    let ck = ContentKey {
        asset_id: 42,
        slice_id: 7,
        representation_id: RepresentationId(1),
        source_gen: 3,
    };

    let sub_a = plane
        .submit(TransferRequest {
            link: LinkId(1),
            key: ScheduleKey::prefetch(GlobalUrgency(0), 5, 1, 0.8, 1),
            bytes: 1024 * 1024,
            content_key: Some(ck),
            placement: PlacementRequirement::AliasInDomain(vram),
            compatibility: BatchCompatibility::InlineSignal,
            src_lease: None,
            dst_lease: make_lease(vram, 10, 1, 1024 * 1024, &pool),
            direct: true,
        })
        .unwrap();
    assert!(!sub_a.joined_existing);

    // Subscriber B joins the same ContentKey with AliasInDomain(vram).
    let sub_b = plane
        .submit(TransferRequest {
            link: LinkId(1),
            key: ScheduleKey::blocking(GlobalUrgency(0), 5, 1, 2),
            bytes: 1024 * 1024,
            content_key: Some(ck),
            placement: PlacementRequirement::AliasInDomain(vram),
            compatibility: BatchCompatibility::InlineSignal,
            src_lease: None,
            dst_lease: make_lease(vram, 11, 1, 1024 * 1024, &pool),
            direct: true,
        })
        .unwrap();
    assert!(sub_b.joined_existing);
    assert_eq!(sub_b.op_id, sub_a.op_id);
    assert_eq!(sub_b.dst_slot, 10, "B receives alias lease to A's slot 10");
    assert!(pool.is_released(11, 1), "B's redundant lease 11 released immediately outside lock");

    // Different representation_id must NOT join!
    let sub_c = plane
        .submit(TransferRequest {
            link: LinkId(1),
            key: ScheduleKey::blocking(GlobalUrgency(0), 5, 1, 3),
            bytes: 1024 * 1024,
            content_key: Some(ContentKey {
                representation_id: RepresentationId(2),
                ..ck
            }),
            placement: PlacementRequirement::AliasInDomain(vram),
            compatibility: BatchCompatibility::InlineSignal,
            src_lease: None,
            dst_lease: make_lease(vram, 12, 1, 1024 * 1024, &pool),
            direct: true,
        })
        .unwrap();
    assert!(!sub_c.joined_existing, "mismatched representation_id must not join");
    drop(sub_c); // cancels immediately while Queued
    assert!(pool.is_released(12, 1));

    // Issue the physical op.
    let batch = plane
        .claim_next_batch(LinkId(1), 64 * 1024 * 1024, 1.0)
        .expect("claimed");
    assert_eq!(batch.ops.len(), 1);
    assert_eq!(
        batch.ops[0].class,
        PriorityClass::Blocking,
        "B's Blocking join promoted A's Prefetch op before issue"
    );
    plane.confirm_issued(batch.batch_id, None);

    // Subscriber A cancels while Issued.
    drop(sub_a);
    assert!(!pool.is_released(10, 1), "slot 10 must stay pinned while Issued");

    // Hardware completes the op.
    plane.retire_issued(batch.batch_id);
    assert!(matches!(sub_b.outcome(), Some(TransferOutcome::Completed { .. })));
    assert!(
        !pool.is_released(10, 1),
        "slot 10 must stay pinned after HW completion until subscriber B detaches"
    );

    // Subscriber B detaches -> slot 10 now releases cleanly.
    drop(sub_b);
    assert!(pool.is_released(10, 1));
}

/// Scenario 3: Mid-Batch Partial Issue Failure.
/// Driver fails after accepting copy 1 of 4 -> `Submitting -> Quarantined`,
/// zero destination slots returned early, backing retained until teardown.
#[test]
fn scenario_3_mid_batch_partial_issue_failure_quarantines_backing() {
    let plane = IoPlane::new(discrete_fabric());
    let pool = TrackingPool::new();
    let vram = DomainId::DeviceVram(DeviceId(0));
    let compat = BatchCompatibility::SharedConsumer { step: 2, layer: 0 };

    let mut handles = Vec::new();
    for slot in 0..4u32 {
        handles.push(
            plane
                .submit(TransferRequest {
                    link: LinkId(1),
                    key: ScheduleKey::blocking(GlobalUrgency(0), 2, 0, slot as u64),
                    bytes: 1024 * 1024,
                    content_key: None,
                    placement: PlacementRequirement::FixedDestination,
                    compatibility: compat,
                    src_lease: None,
                    dst_lease: make_lease(vram, slot, 1, 1024 * 1024, &pool),
                    direct: true,
                })
                .unwrap(),
        );
    }

    let batch = plane
        .claim_next_batch(LinkId(1), 64 * 1024 * 1024, 1.0)
        .expect("4-op batch");
    assert_eq!(batch.ops.len(), 4);

    // Driver fails ambiguously after copy 1 of 4 -> quarantine the batch.
    plane.quarantine_batch(batch.batch_id, "cuMemcpyHtoDAsync failed after item 1/4");

    for (slot, h) in handles.into_iter().enumerate() {
        assert!(matches!(h.outcome(), Some(TransferOutcome::Quarantined(_))));
        assert!(pool.is_quarantined(slot as u32, 1));
        assert!(!pool.is_released(slot as u32, 1));
        drop(h);
        assert!(
            !pool.is_released(slot as u32, 1),
            "quarantined slot {slot} must NEVER be returned to pool on subscriber drop"
        );
    }

    assert_eq!(plane.quarantined_lease_count(), 4);
    let ledger = plane.link_ledger(LinkId(1)).unwrap();
    assert_eq!(ledger.total_in_flight_bytes(), 0);
    assert_eq!(ledger.quarantined_ops, 4);

    // Explicit device teardown releases quarantined leases.
    assert_eq!(plane.teardown_quarantined(), 4);
    for slot in 0..4u32 {
        assert!(pool.is_released(slot, 1));
    }
}

/// Scenario 4: Inverted Producer Completions.
/// Independent streams B and A complete out of order -> per-(DeviceId, ProducerId)
/// `TimelinePoint`s never corrupt or erase each other's fences, and `HostOnly`
/// fences are rejected for GPU device waits.
#[test]
fn scenario_4_inverted_producer_completions_and_host_only_rejection() {
    let tl = MultiProducerTimeline::new();
    let dev = DeviceId(0);
    let stream_a = ProducerId(101);
    let stream_b = ProducerId(202);
    tl.register(dev, stream_a);
    tl.register(dev, stream_b);

    let pt_a1 = TimelinePoint::new(dev, stream_a, 1, FenceCapability::DeviceAndHost);
    let pt_a2 = TimelinePoint::new(dev, stream_a, 2, FenceCapability::DeviceAndHost);
    let pt_b1 = TimelinePoint::new(dev, stream_b, 5, FenceCapability::DeviceAndHost);

    // Stream B completes to 5 before Stream A completes even 1.
    tl.advance(dev, stream_b, 5);
    assert!(tl.is_reached(pt_b1).unwrap());
    assert!(!tl.is_reached(pt_a1).unwrap());
    assert!(!tl.is_reached(pt_a2).unwrap());

    // Stream A faults at 1; Stream B remains unaffected.
    tl.advance(dev, stream_a, 1);
    tl.fault(dev, stream_a, "stream A fault");
    assert!(tl.is_reached(pt_a1).unwrap());
    assert!(matches!(tl.is_reached(pt_a2), Err(FenceError::Faulted(_))));
    assert!(tl.is_reached(pt_b1).unwrap());

    // HostOnly rejected for GPU wait.
    let host_pt = TimelinePoint::new(dev, ProducerId(303), 1, FenceCapability::HostOnly);
    assert_eq!(
        host_pt.validate_for_device_wait(dev),
        Err(FenceError::HostOnlyOnDeviceWait(host_pt))
    );
}

/// Scenario 5: Reader-Retirement Overlap.
/// Slot reuse attempted while a reserved unpublished pass still references the
/// old map -> blocked until reader retirement completes.
#[test]
fn scenario_5_reader_retirement_overlap() {
    let tracker = ReaderRetirementTracker::new();
    let slot = SlotKey {
        domain: DomainId::DeviceVram(DeviceId(0)),
        slot: 5,
    };
    let active = tracker.pin_active_reader(slot, 9);
    let mut reserved = tracker.reserve_unpublished(slot, 9);
    assert!(!tracker.can_overwrite(slot));

    // Active reader retires, but reserved unpublished pass STILL blocks overwrite!
    active.retire();
    assert!(
        !tracker.can_overwrite(slot),
        "reserved unpublished pass must still block slot overwrite"
    );

    // Reserved pass submits and later retires -> overwrite now permitted.
    reserved.mark_submitted();
    assert!(!tracker.can_overwrite(slot));
    reserved.retire();
    assert!(tracker.can_overwrite(slot));
}

/// Scenario 6: 1:1 Physical Debit/Credit Conservation + Hot-Path Latency.
/// Randomized submit/reprice/cancel/rollback/refuse/quarantine/retire fuzzing
/// returns `in_flight_bytes == 0` and `CoupledSet` bus credits == 0 on quiescence.
#[test]
fn scenario_6_conservation_fuzz_and_hot_path_latency() {
    let plane = IoPlane::new(discrete_fabric());
    let pool = TrackingPool::new();
    let vram = DomainId::DeviceVram(DeviceId(0));
    let ram = DomainId::HostRam(NumaNodeId(0));
    let ddr0 = SharedBusId::HostDdr(NumaNodeId(0));
    let pcie0 = SharedBusId::PcieRoot(RootComplexId(0));

    let mut rng: u64 = 0x5E15_1C10_2026_u64;
    let mut next_u64 = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    let mut handles: Vec<seismic_io::TransferHandle> = Vec::new();
    let mut submitting = Vec::new();
    let mut issued = Vec::new();
    let mut next_slot = 1u32;

    for step in 0..2_000u64 {
        let action = next_u64() % 10;
        match action {
            0..=3 => {
                // Submit on Link 0 (NVMe) or Link 1 (H2D)
                let link = if next_u64() % 2 == 0 { LinkId(0) } else { LinkId(1) };
                let dst_domain = if link == LinkId(0) { ram } else { vram };
                let class_roll = next_u64() % 4;
                let key = match class_roll {
                    0 => ScheduleKey::blocking(GlobalUrgency(0), step, 0, step),
                    1 => ScheduleKey::streaming(GlobalUrgency(0), step, 0, step),
                    2 => ScheduleKey::prefetch(GlobalUrgency(0), step + 1, 0, 0.7, step),
                    _ => ScheduleKey::background(GlobalUrgency(0), step),
                };
                let slot = next_slot;
                next_slot += 1;
                let h = plane
                    .submit(TransferRequest {
                        link,
                        key,
                        bytes: 512 * 1024,
                        content_key: None,
                        placement: PlacementRequirement::AliasInDomain(dst_domain),
                        compatibility: BatchCompatibility::InlineSignal,
                        src_lease: None,
                        dst_lease: make_lease(dst_domain, slot, 1, 512 * 1024, &pool),
                        direct: true,
                    })
                    .unwrap();
                handles.push(h);
            }
            4 => {
                // Reprice random handle
                if !handles.is_empty() {
                    let idx = (next_u64() as usize) % handles.len();
                    handles[idx].reprice(ScheduleKey::blocking(GlobalUrgency(0), step, 0, step));
                }
            }
            5 => {
                // Cancel / detach random handle
                if !handles.is_empty() {
                    let idx = (next_u64() as usize) % handles.len();
                    handles.swap_remove(idx);
                }
            }
            6..=7 => {
                // Claim batch on random link
                let link = if next_u64() % 2 == 0 { LinkId(0) } else { LinkId(1) };
                if let Some(batch) = plane.claim_next_batch(link, 4 * 1024 * 1024, 1.0) {
                    submitting.push(batch.batch_id);
                }
            }
            8 => {
                // Advance a Submitting batch (issue, rollback, refuse, or quarantine)
                if !submitting.is_empty() {
                    let idx = (next_u64() as usize) % submitting.len();
                    let bid = submitting.swap_remove(idx);
                    match next_u64() % 10 {
                        0 => plane.rollback_transient(bid),
                        1 => plane.refuse_before_issue(bid, "simulated pre-issue refusal"),
                        2 => plane.quarantine_batch(bid, "simulated partial failure"),
                        _ => {
                            plane.confirm_issued(bid, None);
                            issued.push(bid);
                        }
                    }
                }
            }
            _ => {
                // Retire an Issued batch
                if !issued.is_empty() {
                    let idx = (next_u64() as usize) % issued.len();
                    let bid = issued.swap_remove(idx);
                    plane.retire_issued(bid);
                }
            }
        }
    }

    // Drain everything to quiescence.
    for bid in submitting.drain(..) {
        plane.refuse_before_issue(bid, "drain");
    }
    for bid in issued.drain(..) {
        plane.retire_issued(bid);
    }
    handles.clear();
    let _ = plane.teardown_quarantined();

    for link in [LinkId(0), LinkId(1)] {
        let ledger = plane.link_ledger(link).unwrap();
        assert_eq!(ledger.total_in_flight_bytes(), 0, "link {link:?} in_flight_bytes must return to 0");
        assert_eq!(ledger.total_in_flight_ops(), 0, "link {link:?} in_flight_ops must return to 0");
        assert_eq!(ledger.submitting_bytes, 0);
        assert_eq!(ledger.total_queued_bytes(), 0);
    }
    assert_eq!(plane.bus_ledger(ddr0).unwrap().in_flight_weighted_bytes, 0);
    assert_eq!(plane.bus_ledger(pcie0).unwrap().in_flight_weighted_bytes, 0);
    assert_eq!(pool.released_count(), (next_slot - 1) as usize);

    // Hot-path submit + reprice + cancel microbenchmark sanity check.
    let iters = 10_000u64;
    let start = Instant::now();
    for i in 0..iters {
        let h = plane
            .submit(TransferRequest {
                link: LinkId(1),
                key: ScheduleKey::prefetch(GlobalUrgency(0), i, 0, 0.5, i),
                bytes: 4096,
                content_key: None,
                placement: PlacementRequirement::FixedDestination,
                compatibility: BatchCompatibility::Isolated,
                src_lease: None,
                dst_lease: BufferLease::unmanaged(vram, NonNull::dangling(), 0, 0, 4096, 4096, 1),
                direct: true,
            })
            .unwrap();
        h.reprice(ScheduleKey::blocking(GlobalUrgency(0), i, 0, i));
        drop(h);
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() / (iters as u128);

    // Lock-free seqlock quote() latency check.
    let q_iters = 100_000u64;
    let q_start = Instant::now();
    let mut acc = 0.0f64;
    for _ in 0..q_iters {
        let q = std::hint::black_box(plane.quote(LinkId(1), 2 * 1024 * 1024).unwrap());
        acc += q.estimated_s;
    }
    let q_avg_ns = q_start.elapsed().as_nanos() / (q_iters as u128);
    assert!(acc > 0.0);
    eprintln!(
        "[io_v2_replay] submit+reprice+cancel avg = {avg_ns} ns | seqlock quote() avg = {q_avg_ns} ns"
    );

    // In debug mode allow generous headroom; in release mode enforce the < 500 ns bar.
    let ceiling_ns = if cfg!(debug_assertions) { 50_000 } else { 500 };
    assert!(
        avg_ns < ceiling_ns,
        "submit+reprice+cancel avg {avg_ns} ns exceeded ceiling {ceiling_ns} ns"
    );
}

/// Additional Spec Invariants:
/// - Residual speculative service-time bound (`τ_hol`) prevents 2×8 MiB speculative
///   batches from saturating PCIe and stalling arriving `Blocking` misses.
/// - `Blocking` on NVMe preempts `Prefetch` on H2D when sharing `HostDdr(0)`.
/// - GPU-fence-bound `BusReservation` persists past host scope until `TimelinePoint`
///   is reached and reduces `IoPlane::quote()` `effective_bps`.
/// - `IntermediateSlotReserve` reserves forward-progress slots for `Blocking`.
/// - `PipelineHandle` propagates `reprice(Blocking)` end-to-end.
#[test]
fn residual_service_time_coupled_preemption_fence_reservation_and_pipeline() {
    let plane = IoPlane::new(discrete_fabric());
    let vram = DomainId::DeviceVram(DeviceId(0));
    let ram = DomainId::HostRam(NumaNodeId(0));
    let ddr0 = SharedBusId::HostDdr(NumaNodeId(0));

    // 1. Residual speculative service-time bound on H2D:
    //    tau_hol = max(2*5.1µs, 100µs) = 100µs.
    //    One 4 MiB Prefetch op at 25 GB/s has service time 160 µs > tau_hol.
    //    First speculative op is admitted (idle), second speculative op MUST be
    //    held back by the residual service-time bound!
    let _p1 = plane
        .submit(TransferRequest {
            link: LinkId(1),
            key: ScheduleKey::prefetch(GlobalUrgency(0), 10, 0, 0.9, 1),
            bytes: 4 * 1024 * 1024,
            content_key: None,
            placement: PlacementRequirement::FixedDestination,
            compatibility: BatchCompatibility::InlineSignal,
            src_lease: None,
            dst_lease: BufferLease::unmanaged(vram, NonNull::dangling(), 0, 0, 4 * 1024 * 1024, 4096, 1),
            direct: true,
        })
        .unwrap();
    let _p2 = plane
        .submit(TransferRequest {
            link: LinkId(1),
            key: ScheduleKey::prefetch(GlobalUrgency(0), 10, 1, 0.8, 2),
            bytes: 4 * 1024 * 1024,
            content_key: None,
            placement: PlacementRequirement::FixedDestination,
            compatibility: BatchCompatibility::InlineSignal,
            src_lease: None,
            dst_lease: BufferLease::unmanaged(vram, NonNull::dangling(), 0, 0, 4 * 1024 * 1024, 4096, 1),
            direct: true,
        })
        .unwrap();

    let t0 = Instant::now();
    let b1 = plane
        .claim_next_batch_at(LinkId(1), 16 * 1024 * 1024, 1.0, t0)
        .expect("first speculative op admitted when idle");
    plane.confirm_issued(b1.batch_id, None);
    assert!(
        plane
            .claim_next_batch_at(LinkId(1), 16 * 1024 * 1024, 1.0, t0)
            .is_none(),
        "second speculative op must be blocked by residual service-time bound tau_hol"
    );
    plane.retire_issued_at(b1.batch_id, t0 + Duration::from_micros(200));

    // 2. Coupled-bus Blocking preemption:
    //    - On unsaturated HostDdr (6 + 25 = 31 GB/s < 60 GB/s) and opposite
    //      full-duplex PCIe directions, NVMe Blocking does NOT stall H2D Prefetch.
    //    - Once HostDdr is saturated (e.g., 35 GB/s reservation: 35+6+25 >= 60 GB/s),
    //      NVMe Blocking DOES preempt H2D Prefetch, and clearing NVMe Blocking
    //      invokes the registered cross-link wakeup hook.
    let wakes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let wakes_hook = Arc::clone(&wakes);
    let _hook_guard = plane.register_blocking_cleared_hook(
        LinkId(1),
        Arc::new(move || {
            wakes_hook.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }),
    );
    let nvme_blocking = plane
        .submit(TransferRequest {
            link: LinkId(0),
            key: ScheduleKey::blocking(GlobalUrgency(0), 1, 0, 10),
            bytes: 1024 * 1024,
            content_key: None,
            placement: PlacementRequirement::FixedDestination,
            compatibility: BatchCompatibility::Isolated,
            src_lease: None,
            dst_lease: BufferLease::unmanaged(ram, NonNull::dangling(), 0, 0, 1024 * 1024, 4096, 1),
            direct: true,
        })
        .unwrap();
    assert!(
        !plane.has_coupled_blocking_waiting(LinkId(1)),
        "unsaturated HostDdr must NOT trigger cross-link Blocking preemption"
    );
    {
        let _sat = plane.reserve_bus_scoped(ddr0, 35e9).unwrap();
        assert!(
            plane
                .claim_next_batch_at(LinkId(1), 16 * 1024 * 1024, 1.0, t0 + Duration::from_micros(200))
                .is_none(),
            "H2D Prefetch must yield while NVMe Blocking is queued on saturated HostDdr(0)"
        );
    }
    drop(nvme_blocking);
    assert_eq!(
        wakes.load(std::sync::atomic::Ordering::Acquire),
        1,
        "clearing NVMe Blocking must invoke registered cross-link wakeup hook"
    );

    // 3. GPU-fence-bound BusReservation persists past host drop until TimelinePoint reached.
    let tl = MultiProducerTimeline::new();
    let dev = DeviceId(0);
    let prod = ProducerId(7);
    tl.register(dev, prod);
    let pt = TimelinePoint::new(dev, prod, 1, FenceCapability::DeviceAndHost);

    let baseline_q = plane.quote(LinkId(1), 2 * 1024 * 1024).unwrap();
    {
        let _res_id = plane
            .reserve_bus_until_fence(ddr0, 50e9, pt, Arc::clone(&tl) as Arc<dyn TimelineObserver>)
            .unwrap();
        // `_res_id` drops here, simulating host submit return!
    }
    let contended_q = plane.quote(LinkId(1), 2 * 1024 * 1024).unwrap();
    assert!(
        contended_q.effective_bps < baseline_q.effective_bps,
        "fence-bound bus reservation must remain debited after host submit returns"
    );
    assert!(contended_q.estimated_s > baseline_q.estimated_s);

    tl.advance(dev, prod, 1);
    plane.poll_fence_reservations();
    let restored_q = plane.quote(LinkId(1), 2 * 1024 * 1024).unwrap();
    assert!(
        (restored_q.effective_bps - baseline_q.effective_bps).abs() < 1.0,
        "fence-bound reservation must be credited once TimelinePoint is reached"
    );

    // 4. IntermediateSlotReserve prevents speculative exhaustion of Blocking reserve.
    let reserve = IntermediateSlotReserve::new(4, 2);
    let _g1 = reserve.try_acquire(PriorityClass::Prefetch).expect("slot 1");
    let _g2 = reserve.try_acquire(PriorityClass::Background).expect("slot 2");
    assert!(
        reserve.try_acquire(PriorityClass::Prefetch).is_none(),
        "speculative must not consume the 2-slot Blocking reserve"
    );
    let _g3 = reserve.try_acquire(PriorityClass::Blocking).expect("blocking slot 3");
    let _g4 = reserve.try_acquire(PriorityClass::Blocking).expect("blocking slot 4");

    // 5. PipelineHandle end-to-end priority inheritance.
    let hop0 = plane
        .submit(TransferRequest {
            link: LinkId(0),
            key: ScheduleKey::background(GlobalUrgency(0), 100),
            bytes: 1024 * 1024,
            content_key: None,
            placement: PlacementRequirement::FixedDestination,
            compatibility: BatchCompatibility::Isolated,
            src_lease: None,
            dst_lease: BufferLease::unmanaged(ram, NonNull::dangling(), 0, 0, 1024 * 1024, 4096, 1),
            direct: true,
        })
        .unwrap();
    let pipeline = PipelineHandle::new(vec![hop0]);
    pipeline.reprice(ScheduleKey::blocking(GlobalUrgency(0), 1, 0, 100));
    let claimed = plane.claim_next_batch(LinkId(0), 8 * 1024 * 1024, 1.0).unwrap();
    assert_eq!(claimed.ops[0].class, PriorityClass::Blocking);
    plane.confirm_issued(claimed.batch_id, None);
    plane.retire_issued(claimed.batch_id);
}

/// Stage 3: Real `seismic_io::Reader` storage reads debit and retire on
/// `seismic_io::global_plane()` (`LINK_STORAGE_NVME` + `BUS_HOST_DDR` +
/// `BUS_PCIE_ROOT`), returning to exact `0` in-flight bytes on quiescence,
/// and bidirectional cross-link `Blocking` preemption holds on `default_fabric()`.
#[test]
fn stage3_real_storage_reader_shares_global_io_plane_and_quiesces_cleanly() {
    use seismic_io::{
        global_plane, Buffer, CachePolicy, Range, Reader, Want, BUS_HOST_DDR,
        BUS_PCIE_ROOT, DOMAIN_DEVICE_VRAM, LINK_COPY_H2D, LINK_STORAGE_NVME,
    };
    use std::io::Write;

    let path = std::env::temp_dir().join(format!(
        "seismic_io_v2_stage3_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let payload: Vec<u8> = (0..16384u32).map(|i| (i & 0xFF) as u8).collect();
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&payload).unwrap();
        f.sync_all().unwrap();
    }

    let before_completed = global_plane()
        .link_ledger(LINK_STORAGE_NVME)
        .unwrap()
        .total_completed_bytes();

    let reader = Reader::open(&path, CachePolicy::Cached).unwrap();
    let dest = Arc::new(Buffer::new(payload.len()));
    let handle = unsafe {
        reader
            .request(
                vec![Range {
                    offset: 0,
                    length: payload.len(),
                    into: dest.clone(),
                    at: 0,
                }],
                Want::now(),
            )
            .unwrap()
    };
    let ended = handle.wait();
    assert!(matches!(ended, seismic_io::Ended::Read { ref failed } if failed.is_empty()));
    let read_bytes = unsafe { dest.bytes() };
    assert_eq!(read_bytes, payload.as_slice());

    let after_ledger = global_plane().link_ledger(LINK_STORAGE_NVME).unwrap();
    assert!(
        after_ledger.total_completed_bytes() >= before_completed + payload.len() as u64,
        "global_plane LINK_STORAGE_NVME must account for real Reader storage reads"
    );
    assert_eq!(after_ledger.total_in_flight_bytes(), 0);
    assert_eq!(
        global_plane().bus_ledger(BUS_HOST_DDR).unwrap().in_flight_weighted_bytes,
        0
    );
    assert_eq!(
        global_plane().bus_ledger(BUS_PCIE_ROOT).unwrap().in_flight_weighted_bytes,
        0
    );

    // Bidirectional cross-link check on a fresh default_fabric() IoPlane:
    // - Unsaturated HostDdr (6 + 25 = 31 GB/s < 60 GB/s): H2D Blocking does NOT
    //   stall NVMe read-ahead.
    // - Saturated HostDdr (+35 GB/s reservation): H2D Blocking DOES set
    //   `has_coupled_blocking_waiting(LINK_STORAGE_NVME)`, and dropping it
    //   clears preemption and fires the cross-link wakeup hook.
    let plane = IoPlane::new_default();
    let storage_wakes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let sw = Arc::clone(&storage_wakes);
    let _hook_guard = plane.register_blocking_cleared_hook(
        LINK_STORAGE_NVME,
        Arc::new(move || {
            sw.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }),
    );
    let h2d_blk = plane
        .submit(TransferRequest {
            link: LINK_COPY_H2D,
            key: ScheduleKey::blocking(GlobalUrgency(0), 1, 0, 1),
            bytes: 1024 * 1024,
            content_key: None,
            placement: PlacementRequirement::FixedDestination,
            compatibility: BatchCompatibility::Isolated,
            src_lease: None,
            dst_lease: BufferLease::unmanaged(
                DOMAIN_DEVICE_VRAM,
                NonNull::dangling(),
                0,
                0,
                1024 * 1024,
                256,
                0,
            ),
            direct: true,
        })
        .unwrap();
    assert!(!plane.has_coupled_blocking_waiting(LINK_STORAGE_NVME));
    let _sat = plane.reserve_bus_scoped(BUS_HOST_DDR, 35e9).unwrap();
    assert!(plane.has_coupled_blocking_waiting(LINK_STORAGE_NVME));
    drop(h2d_blk);
    assert!(!plane.has_coupled_blocking_waiting(LINK_STORAGE_NVME));
    assert_eq!(storage_wakes.load(std::sync::atomic::Ordering::Acquire), 1);

    let _ = std::fs::remove_file(&path);
}
