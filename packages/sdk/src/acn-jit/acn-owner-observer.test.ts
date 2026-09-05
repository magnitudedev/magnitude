import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientError from "@effect/platform/HttpClientError"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import {
  AcnHealthResponseSchema,
  AcnInstanceIdSchema,
} from "@magnitudedev/acn-protocol"
import {
  AcnOwnerRecordSchema,
  ProcessGroupAbsent,
  ProcessGroupController,
  ProcessGroupLeaderLive,
  ProcessGroupStopped,
  type AcnOwnerStore,
} from "@magnitudedev/acn-protocol/coordination"
import { Duration, Effect, Fiber, Option, Schema, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { SDK_ACN_TARGET, SDK_VERSION } from "../version"
import { MAGNITUDE_SERVICE_ORIGIN } from "../inference-endpoint"
import { makeAcnOwnerObserver } from "./acn-owner-observer"

const owner = Schema.decodeUnknownSync(AcnOwnerRecordSchema)({
  pid: 42,
  processStartIdentity: "start-1",
  port: 14_000,
})

const health = Schema.decodeUnknownSync(AcnHealthResponseSchema)({
  service: "magnitude-acn",
  version: SDK_VERSION,
  revision: SDK_ACN_TARGET.revision,
  id: AcnInstanceIdSchema.make("owner-1"),
  pid: owner.pid,
  state: { _tag: "Ready" },
})

const owners: AcnOwnerStore = {
  current: Effect.succeed(Option.some(owner)),
  replaceOwner: () => Effect.dieMessage("owner replacement is outside observer tests"),
}

const processes: ProcessGroupController = {
  inspect: (pid) => Effect.succeed(pid === owner.pid
    ? Option.some({ pid: owner.pid, processStartIdentity: owner.processStartIdentity })
    : Option.none()),
  currentProcess: Effect.succeed({
    pid: owner.pid,
    processStartIdentity: owner.processStartIdentity,
  }),
  observe: (group) => Effect.succeed(group.leader.pid === owner.pid
    ? new ProcessGroupLeaderLive({ group })
    : new ProcessGroupAbsent({ group })),
  waitForGroupExit: () => Effect.succeed(false),
  stop: (group) => Effect.succeed(new ProcessGroupStopped({ group })),
}

type HealthStep = (
  request: HttpClientRequest.HttpClientRequest,
) => Effect.Effect<HttpClientResponse.HttpClientResponse, HttpClientError.HttpClientError>

const validHealth: HealthStep = (request) => Effect.succeed(HttpClientResponse.fromWeb(
  request,
  Response.json(Schema.encodeSync(AcnHealthResponseSchema)(health), { status: 200 }),
))

const startingHealth = Schema.decodeUnknownSync(AcnHealthResponseSchema)({
  ...health,
  state: { _tag: "Starting", activity: "Resolving" },
})

const validStartingHealth: HealthStep = (request) => Effect.succeed(HttpClientResponse.fromWeb(
  request,
  Response.json(Schema.encodeSync(AcnHealthResponseSchema)(startingHealth), { status: 503 }),
))

const transportFailure: HealthStep = (request) => Effect.fail(new HttpClientError.RequestError({
  request,
  reason: "Transport",
  description: "simulated local transport failure",
}))

const timeoutFailure: HealthStep = () => Effect.never

const responseDecodeFailure: HealthStep = (request) => Effect.succeed(HttpClientResponse.fromWeb(
  request,
  new Response("{", {
    status: 200,
    headers: { "content-type": "application/json" },
  }),
))

const schemaFailure: HealthStep = (request) => Effect.succeed(HttpClientResponse.fromWeb(
  request,
  Response.json({ service: "not-magnitude" }, { status: 200 }),
))

const failureCases = [
  {
    name: "request transport failure",
    step: transportFailure,
    expectedTag: "AcnHealthRequestFailed",
    clockAdjustments: 0,
  },
  {
    name: "request timeout",
    step: timeoutFailure,
    expectedTag: "AcnHealthAttemptTimedOut",
    clockAdjustments: 1,
  },
  {
    name: "response decoding failure",
    step: responseDecodeFailure,
    expectedTag: "AcnHealthResponseInvalid",
    clockAdjustments: 0,
  },
  {
    name: "health schema failure",
    step: schemaFailure,
    expectedTag: "AcnHealthResponseInvalid",
    clockAdjustments: 0,
  },
] as const

const observeSteps = (
  steps: ReadonlyArray<HealthStep>,
  clockAdjustments: number,
) => Effect.gen(function* () {
  let attempts = 0
  const http = HttpClient.make((request) => Effect.suspend(() => {
    const step = steps[Math.min(attempts, steps.length - 1)]
    attempts += 1
    return step(request)
  }))
  const fiber = yield* makeAcnOwnerObserver(owners, processes, http).observe.pipe(Effect.fork)
  for (let index = 0; index < clockAdjustments; index += 1) {
    yield* Effect.yieldNow()
    yield* TestClock.adjust(Duration.seconds(2))
  }
  return { observation: yield* Fiber.join(fiber), attempts }
})

describe("ACN owner health observation", () => {
  it("checks private owner health but selects the fixed public application endpoint", async () => {
    await Effect.runPromise(Effect.gen(function* () {
      const urls: string[] = []
      const http = HttpClient.make((request) => {
        urls.push(request.url)
        return validHealth(request)
      })
      const observer = makeAcnOwnerObserver(owners, processes, http)
      const observation = yield* observer.observe
      expect(observation._tag).toBe("AcnRecordedOwnerLiveWithHealth")
      if (observation._tag !== "AcnRecordedOwnerLiveWithHealth") return
      const ready = yield* observer.confirmReady(observation.owner, observation.health)
      expect(urls).toEqual([`http://127.0.0.1:${owner.port}/health`])
      expect(Option.getOrThrow(ready)).toMatchObject({
        url: MAGNITUDE_SERVICE_ORIGIN,
        id: health.id,
        pid: owner.pid,
        processStartIdentity: owner.processStartIdentity,
      })
    }))
  })

  it("uses one request when the primary response is valid ready health", async () => {
    const result = await Effect.runPromise(
      observeSteps([validHealth], 0).pipe(Effect.provide(TestContext.TestContext)),
    )

    expect(result.attempts).toBe(1)
    expect(result.observation).toMatchObject({
      _tag: "AcnRecordedOwnerLiveWithHealth",
      health: { status: 200, health },
    })
  })

  it("uses one request when the primary response is valid starting health", async () => {
    const result = await Effect.runPromise(
      observeSteps([validStartingHealth], 0).pipe(Effect.provide(TestContext.TestContext)),
    )

    expect(result.attempts).toBe(1)
    expect(result.observation).toMatchObject({
      _tag: "AcnRecordedOwnerLiveWithHealth",
      health: { status: 503, health: startingHealth },
    })
  })

  for (const failure of failureCases) {
    it(`confirms an isolated ${failure.name} before declaring health unavailable`, async () => {
      const result = await Effect.runPromise(
        observeSteps([failure.step, validHealth], failure.clockAdjustments).pipe(
          Effect.provide(TestContext.TestContext),
        ),
      )

      expect(result.attempts).toBe(2)
      expect(result.observation).toMatchObject({
        _tag: "AcnRecordedOwnerLiveWithHealth",
        owner,
        health: { status: 200, health },
      })
    })

    it(`preserves two ${failure.name} results when health remains unavailable`, async () => {
      const result = await Effect.runPromise(
        observeSteps(
          [failure.step, failure.step],
          failure.clockAdjustments * 2,
        ).pipe(Effect.provide(TestContext.TestContext)),
      )

      expect(result.attempts).toBe(2)
      expect(result.observation).toMatchObject({
        _tag: "AcnRecordedOwnerLiveWithoutHealth",
        owner,
        attempts: [
          { _tag: failure.expectedTag },
          { _tag: failure.expectedTag },
        ],
      })
    })
  }

  it("preserves different first and second failure mechanisms in order", async () => {
    const result = await Effect.runPromise(
      observeSteps([transportFailure, schemaFailure], 0).pipe(
        Effect.provide(TestContext.TestContext),
      ),
    )

    expect(result.attempts).toBe(2)
    expect(result.observation).toMatchObject({
      _tag: "AcnRecordedOwnerLiveWithoutHealth",
      attempts: [
        {
          _tag: "AcnHealthRequestFailed",
          message: expect.stringContaining("simulated local transport failure"),
        },
        { _tag: "AcnHealthResponseInvalid" },
      ],
    })
  })
})
