use magnitude_service::policy::{
    order_victims, AvailabilityEpoch, Operation, Phase, RequestId, Scheduler, ServiceLimits, Victim,
};

fn limits() -> ServiceLimits {
    ServiceLimits {
        max_requests: 8,
        max_batch: 2,
        prefill_tokens: 8,
        decode_tokens: 1,
        decode_share: 0.5,
        locality_seconds: 0.0,
    }
}

fn operation(identity: u64, phase: Phase, waiting_since_ns: u64) -> Operation {
    Operation {
        identity: RequestId(identity),
        phase,
        active: false,
        resident: true,
        waiting_since_ns,
        service_ns: 0,
        preemption_debt: 0,
    }
}

#[test]
fn contended_phase_share_is_charged_by_completed_service_time() {
    let mut scheduler = Scheduler::new(limits()).unwrap();
    let candidates = [
        operation(1, Phase::Prefill, 0),
        operation(2, Phase::Decode, 0),
    ];

    let first = scheduler.select(&candidates, 10).unwrap().unwrap();
    assert_eq!(first.phase(), Phase::Decode);
    scheduler.completed(first, 20);

    let second = scheduler.select(&candidates, 30).unwrap().unwrap();
    assert_eq!(second.phase(), Phase::Prefill);
    scheduler.completed(second, 20);

    let third = scheduler.select(&candidates, 50).unwrap().unwrap();
    assert_eq!(third.phase(), Phase::Decode);
    assert_eq!(scheduler.completed_service_ns(), 40);
}

#[test]
fn fairness_is_deterministic_and_never_exceeds_batch_capacity() {
    let mut scheduler = Scheduler::new(limits()).unwrap();
    let candidates = [
        operation(3, Phase::Decode, 9),
        operation(1, Phase::Decode, 0),
        operation(2, Phase::Decode, 5),
    ];
    let selection = scheduler.select(&candidates, 10).unwrap().unwrap();
    assert_eq!(selection.requests(), &[RequestId(1), RequestId(2)]);
}

#[test]
fn victim_order_prioritizes_blocked_output_then_actual_replay_price() {
    let mut victims = [
        Victim {
            identity: RequestId(1),
            output_blocked: false,
            preemption_debt: 0,
            exclusive_bytes: 100,
            replay_tokens: 10,
            service_ns: 0,
        },
        Victim {
            identity: RequestId(2),
            output_blocked: true,
            preemption_debt: 0,
            exclusive_bytes: 1,
            replay_tokens: 100,
            service_ns: 0,
        },
        Victim {
            identity: RequestId(3),
            output_blocked: false,
            preemption_debt: 0,
            exclusive_bytes: 20,
            replay_tokens: 1,
            service_ns: 0,
        },
    ];
    order_victims(&mut victims);
    assert_eq!(
        victims.map(|victim| victim.identity),
        [RequestId(2), RequestId(3), RequestId(1)]
    );
}

#[test]
fn availability_reconsideration_is_epoch_gated() {
    let blocked = AvailabilityEpoch::default();
    let mut current = blocked;
    assert!(!current.changed_since(blocked));
    current.advance().unwrap();
    assert!(current.changed_since(blocked));
}
