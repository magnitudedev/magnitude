"""Explicit resource probes use the same durable dispatch and delivery as measurements."""

import time

from .contracts import Attempt, Experiment, Request, identity, now
from .sources import capture
from .store import Store
from .transport import ensure_service


def submit(config, target_name):
    target = config.targets[target_name]
    with Store(config.workspace) as store:
        source = capture(config.root, store)
        request = Request(
            request_id=identity(),
            created=now(),
            model=None,
            operation="characterize",
            experiment=Experiment(
                model="resources", source=source.source_id, targets=(target_name,)
            ),
            attempts=(Attempt(attempt_id=identity(), target_name=target_name, target=target),),
        )
        store.put("request", request.request_id, request)
    ensure_service(config.workspace, config.root, "coordinator")
    from .service import TERMINAL

    while True:
        with Store(config.workspace, readonly=True) as store:
            request = store.request(request.request_id)
        if all(a.status in TERMINAL for a in request.attempts):
            return request.model_dump()
        time.sleep(0.5)


def execute(attempt, store):
    import ops
    from engine import DevicePlan
    from ops.lab.characterization import ProbeProtocol, characterize
    from ops.lab.store import ObservationStore

    device = ops.DeviceRuntime.open(
        DevicePlan.discover(
            backend=attempt.target.device.backend,
            ordinal=attempt.target.device.index,
            maximum_bytes=attempt.target.device.maximum_bytes or (12 << 30),
        )
    )
    try:
        with ObservationStore(store.root / "ops.sqlite") as native:
            profile = characterize(
                device, native, protocol=ProbeProtocol.for_capacity(device.available_bytes)
            )
            observations = {
                rate.measurement: native.measurement(rate.measurement).model_dump(mode="json")
                for rate in profile.rates
            }
        artifact = store.put_blob(
            __import__("json")
            .dumps({"profile": profile.model_dump(mode="json"), "observations": observations})
            .encode()
        )
        store.put(
            "characterization",
            profile.key,
            {"artifact_id": artifact, "profile": profile.model_dump(mode="json")},
            mutable=True,
        )
        return artifact
    finally:
        device.close()
