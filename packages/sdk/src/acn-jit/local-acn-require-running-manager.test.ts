import { describe, expect, it } from "vitest"
import { Effect, Option, Ref, Schema, Stream } from "effect"
import {
  AcnHealthResponseSchema,
  AcnReadyInstanceSchema,
} from "@magnitudedev/acn-protocol"
import { AcnOwnerRecordSchema } from "@magnitudedev/acn-protocol/coordination"
import { SDK_ACN_TARGET } from "../version"
import { MAGNITUDE_SERVICE_ORIGIN } from "../inference-endpoint"
import { runAcnEnsure } from "./acn-instance-manager"
import {
  AcnRecordedOwnerAbsent,
  AcnRecordedOwnerLiveWithoutHealth,
  AcnRecordedOwnerLiveWithHealth,
  type AcnOwnerObserver,
} from "./acn-owner-observer"
import { AcnHealthAttemptTimedOut } from "./errors"
import {
  makeRequireRunningAcnInstanceManagerFromObserver,
  makeStartingAcnInstanceManagerFromObserver,
} from "./local-acn-require-running-manager"

const unavailableHealth = [
  new AcnHealthAttemptTimedOut({}),
  new AcnHealthAttemptTimedOut({}),
] as const

describe("require-running ACN manager", () => {
  it("fails absent service with the actionable command and performs no bootstrap", async () => {
    const observer: AcnOwnerObserver = {
      observe: Effect.succeed(new AcnRecordedOwnerAbsent({ expectedOwner: Option.none() })),
      confirmReady: () => Effect.succeed(Option.none()),
    }
    const manager = makeRequireRunningAcnInstanceManagerFromObserver(observer)
    const result = await Effect.runPromise(Effect.exit(Effect.scoped(
      runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })),
    )))
    expect(result).toMatchObject({
      _tag: "Failure",
      cause: {
        _tag: "Fail",
        error: {
          _tag: "AcnEnsuranceFailed",
          reason: "Magnitude service is not running. Run `magnitude service start`.",
        },
      },
    })
  })

  it("adopts a newer ready service without emitting startup observations", async () => {
    const owner = Schema.decodeUnknownSync(AcnOwnerRecordSchema)({
      pid: 42,
      processStartIdentity: "start-1",
      port: 14000,
    })
    const health = Schema.decodeUnknownSync(AcnHealthResponseSchema)({
      service: "magnitude-acn",
      id: "instance-1",
      pid: owner.pid,
      version: "9.0.0",
      revision: SDK_ACN_TARGET.revision + 1,
      state: { _tag: "Ready" },
    })
    const ready = Schema.decodeUnknownSync(AcnReadyInstanceSchema)({
      revision: health.revision,
      id: "instance-1",
      identity: health.version,
      url: MAGNITUDE_SERVICE_ORIGIN,
      pid: owner.pid,
      processStartIdentity: owner.processStartIdentity,
      lifecycle: { _tag: "Ready" },
    })
    const observer: AcnOwnerObserver = {
      observe: Effect.succeed(new AcnRecordedOwnerLiveWithHealth({
        owner,
        health: {
          status: 200,
          health,
        },
      })),
      confirmReady: () => Effect.succeed(Option.some(ready)),
    }
    const events = await Effect.runPromise(Effect.scoped(
      Stream.runCollect(makeRequireRunningAcnInstanceManagerFromObserver(observer).ensure({
        target: SDK_ACN_TARGET,
      })),
    ))
    expect(Array.from(events).map((event) => event._tag)).toEqual(["Ready"])
  })

  it("fails an unresponsive live service instead of waiting or bootstrapping", async () => {
    const owner = Schema.decodeUnknownSync(AcnOwnerRecordSchema)({
      pid: 42,
      processStartIdentity: "start-1",
      port: 14000,
    })
    const observer: AcnOwnerObserver = {
      observe: Effect.succeed(new AcnRecordedOwnerLiveWithoutHealth({
        owner,
        attempts: unavailableHealth,
      })),
      confirmReady: () => Effect.succeed(Option.none()),
    }
    const manager = makeRequireRunningAcnInstanceManagerFromObserver(observer)
    const result = await Effect.runPromise(Effect.exit(Effect.scoped(
      runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })),
    )))
    expect(result).toMatchObject({
      _tag: "Failure",
      cause: {
        _tag: "Fail",
        error: {
          _tag: "AcnHealthUnavailable",
          owner,
          attempts: unavailableHealth,
        },
      },
    })
  })

  it("waits through the manager's pre-health interval when service start owns convergence", async () => {
    const owner = Schema.decodeUnknownSync(AcnOwnerRecordSchema)({
      pid: 42,
      processStartIdentity: "start-1",
      port: 14000,
    })
    const health = Schema.decodeUnknownSync(AcnHealthResponseSchema)({
      service: "magnitude-acn",
      id: "instance-1",
      pid: owner.pid,
      version: "9.0.0",
      revision: SDK_ACN_TARGET.revision,
      state: { _tag: "Ready" },
    })
    const ready = Schema.decodeUnknownSync(AcnReadyInstanceSchema)({
      revision: health.revision,
      id: health.id,
      identity: health.version,
      url: MAGNITUDE_SERVICE_ORIGIN,
      pid: owner.pid,
      processStartIdentity: owner.processStartIdentity,
      lifecycle: { _tag: "Ready" },
    })
    const observations = await Effect.runPromise(Ref.make(0))
    const observer: AcnOwnerObserver = {
      observe: Ref.getAndUpdate(observations, (value) => value + 1).pipe(
        Effect.map((value) => value === 0
          ? new AcnRecordedOwnerLiveWithoutHealth({ owner, attempts: unavailableHealth })
          : new AcnRecordedOwnerLiveWithHealth({ owner, health: { status: 200, health } })),
      ),
      confirmReady: () => Effect.succeed(Option.some(ready)),
    }
    const events = await Effect.runPromise(Effect.scoped(
      Stream.runCollect(makeStartingAcnInstanceManagerFromObserver(observer).ensure({
        target: SDK_ACN_TARGET,
      })),
    ))
    expect(Array.from(events).map((event) => event._tag)).toEqual(["Ready"])
  })
})
