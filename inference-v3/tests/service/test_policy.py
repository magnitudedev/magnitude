from engine.service.policy import Operation, Limits, Phase, RequestId, Scheduler


def test_locality_aging_and_completed_physical_service():
    scheduler = Scheduler(Limits(max_batch=2, locality_seconds=0.05))
    resident = Operation(RequestId(1), Phase.PREFILL, True, True, 0, 200, 0)
    queued = Operation(RequestId(0), Phase.PREFILL, False, False, 0, 0, 0)
    selected = scheduler.select((queued, resident), 0)
    assert selected is not None and selected.requests == (resident.identity, queued.identity)
    # A productive resident has just received service; an older peer's waiting
    # age can outweigh both locality preferences without a fixed wait promise.
    resident = Operation(RequestId(1), Phase.PREFILL, True, True, 200_000_000, 300, 0)
    selected = scheduler.select((queued, resident), 200_000_000)
    assert selected is not None and selected.requests[0] == queued.identity
    decode = Operation(RequestId(2), Phase.DECODE, True, True, 0, 0, 0)
    selected = scheduler.select((queued, decode), 0)
    assert selected is not None and selected.phase == Phase.DECODE
    scheduler.completed(selected, 10)
    selected = scheduler.select((queued, decode), 10)
    assert selected is not None and selected.phase == Phase.PREFILL
    scheduler.completed(selected, 40)
    selected = scheduler.select((queued, decode), 50)
    assert selected is not None and selected.phase == Phase.DECODE
    scheduler.completed(selected, 40)
    assert scheduler.completed_service_ns == 90 and scheduler.decode_debt_ns == 0
