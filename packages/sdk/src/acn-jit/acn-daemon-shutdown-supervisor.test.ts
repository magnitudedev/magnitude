import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { ProcessStartIdentitySchema } from "@magnitudedev/acn-protocol"
import {
  AcnOwnerStore,
  ProcessGroupLeaderLive,
  ProcessGroupSignalPermissionDenied,
  ProcessGroupStopped,
  type AcnOwnerRecord,
  type ProcessGroupController,
} from "@magnitudedev/acn-protocol/coordination"
import { Effect, Option, Ref } from "effect"
import { describe, expect, it } from "vitest"
import { makeAcnDaemonShutdownSupervisor } from "./acn-daemon-shutdown-supervisor"

const expected: AcnOwnerRecord = {
  pid: 41_001,
  processStartIdentity: ProcessStartIdentitySchema.make("darwin:test-session:expected"),
  port: 49_152,
}
const group = { leader: { pid: expected.pid, processStartIdentity: expected.processStartIdentity } }
const makeOwnerStore = (owner: Ref.Ref<Option.Option<AcnOwnerRecord>>): AcnOwnerStore =>
  AcnOwnerStore.of({
    current: Ref.get(owner),
    replaceOwner: () => Effect.dieMessage("shutdown supervisor must not replace owners"),
  })
/** A live daemon that ignores graceful shutdown, so the supervisor must escalate to `stop`. */
const liveDaemon = (stop: ProcessGroupController["stop"]): ProcessGroupController => ({
  inspect: () => Effect.succeed(Option.some(group.leader)),
  currentProcess: Effect.succeed(group.leader),
  observe: (target) => Effect.succeed(new ProcessGroupLeaderLive({ group: target })),
  waitForGroupExit: () => Effect.succeed(false),
  stop,
})

describe("AcnDaemonShutdownSupervisor", () => {
  it("surfaces a stop permission denial as the typed shutdown failure", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const owner = yield* Ref.make(Option.some(expected))
      let shutdownRequests = 0
      const http = HttpClient.make((request) => Effect.sync(() => {
        shutdownRequests += 1
        return HttpClientResponse.fromWeb(request, Response.json({}))
      }))
      const processes = liveDaemon(() => Effect.fail(new ProcessGroupSignalPermissionDenied({
        group,
        message: "SystemError: kill() failed: EPERM: Operation not permitted",
      })))
      const supervisor = yield* makeAcnDaemonShutdownSupervisor(makeOwnerStore(owner), processes, http)
      const result = yield* supervisor.shutdown(expected, "HealthUnavailable").pipe(Effect.either)
      expect(shutdownRequests).toBe(1)
      expect(result).toMatchObject({
        _tag: "Left",
        left: {
          _tag: "AcnDaemonShutdownFailed",
          owner: expected,
          reason: "HealthUnavailable",
          failure: {
            _tag: "ProcessGroupSignalPermissionDenied",
            message: "SystemError: kill() failed: EPERM: Operation not permitted",
          },
        },
      })
    }))
  })

  it("does not stop the group after the complete owner row changes", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const owner = yield* Ref.make(Option.some(expected))
      let stops = 0
      const replacement = { ...expected, port: expected.port + 1 }
      const http = HttpClient.make((request) => Ref.set(owner, Option.some(replacement)).pipe(
        Effect.as(HttpClientResponse.fromWeb(request, Response.json({}))),
      ))
      const processes = liveDaemon((target) => Effect.sync(() => {
        stops += 1
        return new ProcessGroupStopped({ group: target })
      }))
      const supervisor = yield* makeAcnDaemonShutdownSupervisor(makeOwnerStore(owner), processes, http)
      const result = yield* supervisor.shutdown(expected, "HealthUnavailable")
      expect(result).toMatchObject({
        _tag: "AcnDaemonSuperseded",
        cause: "OwnerChanged",
        reason: "HealthUnavailable",
      })
      expect(stops).toBe(0)
    }))
  })
})
