use seismic_engine::service::policy::*;
fn scheduler() -> Scheduler {
    Scheduler::new(Limits {
        max_requests: 128,
        max_batch: 2,
        prefill_tokens: 512,
        decode_tokens: 32,
        decode_share: 0.5,
        locality_seconds: 0.0,
    })
    .unwrap()
}
fn op(id: u64, phase: Phase) -> Operation {
    Operation {
        identity: RequestId(id),
        phase,
        active: false,
        resident: true,
        waiting_since_ns: 0,
        service_ns: 0,
        preemption_debt: 0,
    }
}
#[test]
fn contention_starts_with_decode_and_charges_elapsed_time_not_tokens() {
    let mut scheduler = scheduler();
    let candidates = [op(1, Phase::Prefill), op(2, Phase::Decode)];
    let first = scheduler.select(&candidates, 100).unwrap().unwrap();
    assert_eq!(first.phase(), Phase::Decode);
    scheduler.completed(first, 20);
    let prefill = scheduler.select(&candidates, 120).unwrap().unwrap();
    assert_eq!(prefill.phase(), Phase::Prefill);
    scheduler.completed(prefill, 100);
    for now in [220, 240, 260, 280, 300] {
        let decode = scheduler.select(&candidates, now).unwrap().unwrap();
        assert_eq!(decode.phase(), Phase::Decode);
        scheduler.completed(decode, 20);
    }
    assert_eq!(
        scheduler.select(&candidates, 320).unwrap().unwrap().phase(),
        Phase::Prefill
    );
    assert_eq!(scheduler.completed_service_ns(), 220);
    assert!(scheduler.select(&[], 320).unwrap().is_none());
    assert_eq!(
        scheduler.select(&candidates, 320).unwrap().unwrap().phase(),
        Phase::Decode
    );
}
#[test]
fn ranking_uses_waiting_then_service_and_rejects_duplicate_requests() {
    let mut scheduler = scheduler();
    let mut candidates = [
        op(1, Phase::Decode),
        op(2, Phase::Decode),
        op(3, Phase::Decode),
    ];
    candidates[0].waiting_since_ns = 10;
    candidates[1].service_ns = 20;
    let selected = scheduler.select(&candidates, 100).unwrap().unwrap();
    assert_eq!(selected.requests(), &[RequestId(3), RequestId(2)]);
    assert!(scheduler
        .select(&[op(1, Phase::Decode), op(1, Phase::Decode)], 100)
        .is_err());
}
#[test]
fn eviction_prioritizes_blocked_output_then_debt_and_exclusive_replay_price() {
    let mut victims = vec![
        Victim {
            identity: RequestId(1),
            output_blocked: false,
            preemption_debt: 0,
            exclusive_bytes: 1000,
            replay_tokens: 1,
            service_ns: 0,
        },
        Victim {
            identity: RequestId(2),
            output_blocked: true,
            preemption_debt: 1,
            exclusive_bytes: 1000,
            replay_tokens: 1,
            service_ns: 0,
        },
        Victim {
            identity: RequestId(3),
            output_blocked: true,
            preemption_debt: 0,
            exclusive_bytes: 10,
            replay_tokens: 10,
            service_ns: 0,
        },
        Victim {
            identity: RequestId(4),
            output_blocked: true,
            preemption_debt: 0,
            exclusive_bytes: 100,
            replay_tokens: 10,
            service_ns: 0,
        },
    ];
    order_victims(&mut victims);
    assert_eq!(
        victims.iter().map(|v| v.identity).collect::<Vec<_>>(),
        [RequestId(4), RequestId(3), RequestId(2), RequestId(1)]
    );
    let mut epoch = CapacityEpoch::default();
    let blocked = epoch;
    assert!(!epoch.changed_since(blocked));
    epoch.advance().unwrap();
    assert!(epoch.changed_since(blocked));
}
