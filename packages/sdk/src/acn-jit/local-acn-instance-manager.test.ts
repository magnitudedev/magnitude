import * as FileSystem from "@effect/platform/FileSystem"
import { FetchHttpClient } from "@effect/platform"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientResponse from "@effect/platform/HttpClientResponse"
import { BunContext } from "@effect/platform-bun"
import {
  AcnIdentitySchema,
  AcnHealthResponseSchema,
  AcnInstanceIdSchema,
  AcnReady,
  AcnRevisionSchema,
  AcnStarting,
  AcnStopping,
  type AcnHealthState,
  type AcnInstanceId,
  type AcnTarget,
} from "@magnitudedev/acn-protocol"
import {
  ProcessGroupAbsent,
  ProcessGroupController,
  ProcessGroupControllerLive,
  ProcessGroupLeaderLive,
  ProcessGroupStopped,
  makeAcnOwnerStore,
} from "@magnitudedev/acn-protocol/coordination"
import { Duration, Effect, Exit, Fiber, Layer, Option, Schema, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { runAcnEnsure } from "./acn-instance-manager"
import { ChildProcessSpawner } from "./child-process"
import {
  AcnCandidateParentChannelReleaseFailed,
  AcnCandidateSpawnFailed,
} from "./errors"
import {
  makeLocalAcnInstanceManager,
  makeLocalAcnInstanceManagerWithProcessController,
} from "./local-acn-instance-manager"
import { BunSqliteDriverLayer } from "@magnitudedev/acn-protocol/coordination/bun"
import { SDK_ACN_TARGET, SDK_VERSION } from "../version"

const platform = Layer.mergeAll(BunContext.layer, FetchHttpClient.layer, BunSqliteDriverLayer)

const makeExactProcessFixture = Effect.gen(function* () {
  const exact = yield* ProcessGroupControllerLive.currentProcess
  let live = true
  const stop = () => { live = false }
  const controller: ProcessGroupController = {
    inspect: (pid) => Effect.succeed(pid === exact.pid && live ? Option.some(exact) : Option.none()),
    currentProcess: Effect.succeed(exact),
    observe: (group) => Effect.succeed(live
      ? new ProcessGroupLeaderLive({ group })
      : new ProcessGroupAbsent({ group })),
    waitForGroupExit: () => Effect.succeed(!live),
    stop: (group) => Effect.sync(() => {
      stop()
      return new ProcessGroupStopped({ group })
    }),
  }
  return { controller, exact, stop }
})

const makeOwnerHttp = (
  owner: { readonly pid: number },
  id: AcnInstanceId,
  health: () => Option.Option<{ readonly status: number; readonly state: AcnHealthState }>,
  stopOwner: () => void,
  revision = SDK_ACN_TARGET.revision,
) => {
  const requests = { health: 0, shutdown: 0 }
  const client = HttpClient.make((request) => Effect.sync(() => {
    if (request.method === "POST") {
      requests.shutdown += 1
      stopOwner()
      return HttpClientResponse.fromWeb(request, Response.json({}))
    }
    requests.health += 1
    const observed = health()
    if (Option.isNone(observed)) {
      return HttpClientResponse.fromWeb(request, new Response("not health json", { status: 503 }))
    }
    const response = Schema.encodeSync(AcnHealthResponseSchema)({
      service: "magnitude-acn",
      version: SDK_VERSION,
      revision,
      id,
      pid: owner.pid,
      state: observed.value.state,
    })
    return HttpClientResponse.fromWeb(request, new Response(JSON.stringify(response), {
      status: observed.value.status,
      headers: { "content-type": "application/json" },
    }))
  }))
  return { client, requests }
}

describe("LocalAcnInstanceManager", () => {
  it("adopts a newer live owner without resolving or spawning the caller's older artifact", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-adopt-" })
      const requested: AcnTarget = {
        revision: AcnRevisionSchema.make(1_000_000),
        identity: AcnIdentitySchema.make("1.0.0"),
      }
      const selected = AcnRevisionSchema.make(2_000_000)
      const identity = AcnIdentitySchema.make("2.0.0")
      const exact = yield* ProcessGroupControllerLive.currentProcess
      const owners = yield* makeAcnOwnerStore(dataDir)
      const id = AcnInstanceIdSchema.make("selected-owner")
      const server = Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        fetch: () => Response.json({
          service: "magnitude-acn",
          version: identity,
          revision: selected,
          id,
          pid: exact.pid,
          state: new AcnReady({}),
        }),
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))
      if (server.port === undefined) return yield* Effect.dieMessage("test server has no TCP port")
      yield* owners.replaceOwner(Option.none(), { ...exact, port: server.port })

      const manager = yield* makeLocalAcnInstanceManager({
        dataDir,
        binaryPath: `${dataDir}/must-not-be-resolved`,
      }).pipe(Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
        spawn: () => Effect.dieMessage("a valid newer live owner must not spawn"),
      })))
      const ready = yield* runAcnEnsure(manager.ensure({ target: requested }))
      expect(ready.id).toBe(id)
      expect(ready.revision).toBe(selected)
    }).pipe(Effect.provide(platform))))
  })

  it("does not adopt a ready owner below the requested revision", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-upgrade-" })
      const selected = AcnRevisionSchema.make(SDK_ACN_TARGET.revision - 1)
      const exact = yield* ProcessGroupControllerLive.currentProcess
      const server = Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        fetch: () => Response.json({
          service: "magnitude-acn",
          version: SDK_VERSION,
          revision: selected,
          id: AcnInstanceIdSchema.make("obsolete-owner"),
          pid: exact.pid,
          state: new AcnReady({}),
        }),
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))
      if (server.port === undefined) return yield* Effect.dieMessage("test server has no TCP port")
      const owners = yield* makeAcnOwnerStore(dataDir)
      yield* owners.replaceOwner(Option.none(), { ...exact, port: server.port })
      const manager = yield* makeLocalAcnInstanceManager({
        dataDir,
        binaryPath: `${dataDir}/missing-acn`,
      }).pipe(Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
        spawn: () => Effect.dieMessage("replacement preparation must fail before spawning"),
      })))

      expect(Exit.isFailure(yield* Effect.exit(
        runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })),
      ))).toBe(true)
    }).pipe(Effect.provide(platform))))
  })

  it("prepares replacement before retiring a lower live owner", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-replace-lower-" })
      const processFixture = yield* makeExactProcessFixture
      const lowerRevision = AcnRevisionSchema.make(SDK_ACN_TARGET.revision - 1)
      const http = makeOwnerHttp(
        processFixture.exact,
        AcnInstanceIdSchema.make("lower-owner"),
        () => Option.some({ status: 200, state: new AcnReady({}) }),
        processFixture.stop,
        lowerRevision,
      )
      const owners = yield* makeAcnOwnerStore(dataDir)
      yield* owners.replaceOwner(Option.none(), { ...processFixture.exact, port: 49152 })
      let spawns = 0
      const manager = yield* makeLocalAcnInstanceManagerWithProcessController({
        dataDir,
        launchOverride: { target: SDK_ACN_TARGET, command: ["unused-test-acn"] },
      }).pipe(
        Effect.provideService(HttpClient.HttpClient, http.client),
        Effect.provideService(ProcessGroupController, processFixture.controller),
        Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
          spawn: () => Effect.sync(() => { spawns += 1 }).pipe(Effect.zipRight(Effect.never)),
        })),
      )

      const ensuring = yield* runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })).pipe(Effect.fork)
      while (http.requests.shutdown === 0) yield* Effect.yieldNow()
      while (spawns === 0) yield* Effect.yieldNow()
      expect(http.requests.shutdown).toBe(1)
      expect(spawns).toBe(1)
      yield* Fiber.interrupt(ensuring)
    }).pipe(Effect.provide(platform))))
  })

  it("replaces a stale owner because its dead revision has no authority", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-launch-" })
      const exact = yield* ProcessGroupControllerLive.currentProcess
      const id = AcnInstanceIdSchema.make("launched-owner")
      const server = Bun.serve({
        hostname: "127.0.0.1",
        port: 0,
        fetch: () => Response.json({
          service: "magnitude-acn",
          version: SDK_VERSION,
          revision: SDK_ACN_TARGET.revision,
          id,
          pid: exact.pid,
          state: new AcnReady({}),
        }),
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))
      if (server.port === undefined) return yield* Effect.dieMessage("test server has no TCP port")
      const owners = yield* makeAcnOwnerStore(dataDir)
      yield* owners.replaceOwner(Option.none(), {
        pid: 9_000_001,
        processStartIdentity: exact.processStartIdentity,
        port: 49_151,
      })
      let spawns = 0
      const spawner = ChildProcessSpawner.of({
        spawn: () => Effect.gen(function* () {
          spawns += 1
          const expected = yield* owners.current
          const replaced = yield* owners.replaceOwner(
            expected,
            { ...exact, port: server.port! },
          )
          if (replaced._tag !== "Replaced") return yield* Effect.dieMessage("candidate was not admitted")
          return {
            pid: exact.pid,
            exited: Effect.never,
            confirmExactProcess: () => Effect.void,
            admit: Effect.void,
            stopAndReap: Effect.void,
          }
        }).pipe(Effect.mapError((error) => new AcnCandidateSpawnFailed({ message: String(error) }))),
      })
      const manager = yield* makeLocalAcnInstanceManager({
        dataDir,
        launchOverride: {
          target: SDK_ACN_TARGET,
          command: ["unused-test-acn"],
        },
      }).pipe(Effect.provideService(ChildProcessSpawner, spawner))

      const ready = yield* runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET }))
      expect(ready.id).toBe(id)
      expect(spawns).toBe(1)
    }).pipe(Effect.provide(platform))))
  })

  it("reports retained stderr when its admitted daemon exits before readiness", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-admitted-exit-" })
      const processFixture = yield* makeExactProcessFixture
      const owners = yield* makeAcnOwnerStore(dataDir)
      let admissions = 0
      const manager = yield* makeLocalAcnInstanceManagerWithProcessController({
        dataDir,
        launchOverride: { target: SDK_ACN_TARGET, command: ["unused-test-acn"] },
      }).pipe(
        Effect.provideService(ProcessGroupController, processFixture.controller),
        Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
          spawn: () => Effect.gen(function* () {
            const expected = yield* owners.current
            const replaced = yield* owners.replaceOwner(
              expected,
              { ...processFixture.exact, port: 49_152 },
            )
            if (replaced._tag !== "Replaced") {
              return yield* Effect.dieMessage("candidate was not admitted")
            }
            return {
              pid: processFixture.exact.pid,
              exited: Effect.succeed({ code: 17, stderr: "fatal startup detail" }),
              confirmExactProcess: () => Effect.void,
              admit: Effect.sync(() => {
                admissions += 1
                processFixture.stop()
              }),
              stopAndReap: Effect.void,
            }
          }).pipe(Effect.mapError((error) => new AcnCandidateSpawnFailed({ message: String(error) }))),
        })),
      )

      const result = yield* runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })).pipe(Effect.either)
      expect(result).toMatchObject({
        _tag: "Left",
        left: {
          _tag: "AcnCandidateExitedAfterAdmission",
          pid: processFixture.exact.pid,
          code: 17,
          stderr: "fatal startup detail",
        },
      })
      expect(admissions).toBe(1)
    }).pipe(Effect.provide(platform))))
  })

  it("reports an admitted daemon exit after another owner replaces its row", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-replaced-owner-exit-" })
      const current = yield* ProcessGroupControllerLive.currentProcess
      const candidate = current
      const successor = { ...current, pid: current.pid + 1 }
      const owners = yield* makeAcnOwnerStore(dataDir)
      const successorId = AcnInstanceIdSchema.make("successor-owner")
      const controller: ProcessGroupController = {
        inspect: (pid) => Effect.succeed(
          pid === candidate.pid || pid === successor.pid
            ? Option.some({ pid, processStartIdentity: current.processStartIdentity })
            : Option.none(),
        ),
        currentProcess: Effect.succeed(current),
        observe: (group) => Effect.succeed(new ProcessGroupLeaderLive({ group })),
        waitForGroupExit: () => Effect.succeed(false),
        stop: (group) => Effect.succeed(new ProcessGroupStopped({ group })),
      }
      const http = HttpClient.make((request) => Effect.succeed(
        request.url.includes(":49153/")
          ? HttpClientResponse.fromWeb(request, Response.json(
              Schema.encodeSync(AcnHealthResponseSchema)({
                service: "magnitude-acn",
                version: SDK_VERSION,
                revision: SDK_ACN_TARGET.revision,
                id: successorId,
                pid: successor.pid,
                state: new AcnReady({}),
              }),
            ))
          : HttpClientResponse.fromWeb(request, new Response("starting", { status: 503 })),
      ))
      const manager = yield* makeLocalAcnInstanceManagerWithProcessController({
        dataDir,
        launchOverride: { target: SDK_ACN_TARGET, command: ["unused-test-acn"] },
      }).pipe(
        Effect.provideService(HttpClient.HttpClient, http),
        Effect.provideService(ProcessGroupController, controller),
        Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
          spawn: () => Effect.gen(function* () {
            const replaced = yield* owners.replaceOwner(
              yield* owners.current,
              { ...candidate, port: 49_152 },
            )
            if (replaced._tag !== "Replaced") {
              return yield* Effect.dieMessage("candidate was not admitted")
            }
            return {
              pid: candidate.pid,
              exited: Effect.succeed({ code: 23, stderr: "original daemon failed" }),
              confirmExactProcess: () => Effect.void,
              admit: owners.replaceOwner(
                Option.some({ ...candidate, port: 49_152 }),
                { ...successor, port: 49_153 },
              ).pipe(
                Effect.mapError((error) => new AcnCandidateParentChannelReleaseFailed({
                  pid: candidate.pid,
                  message: String(error),
                })),
                Effect.flatMap((result) => result._tag === "Replaced"
                  ? Effect.void
                  : Effect.dieMessage("successor did not replace admitted candidate")),
              ),
              stopAndReap: Effect.void,
            }
          }).pipe(Effect.mapError((error) => new AcnCandidateSpawnFailed({ message: String(error) }))),
        })),
      )

      const result = yield* runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })).pipe(Effect.either)
      expect(result).toMatchObject({
        _tag: "Left",
        left: {
          _tag: "AcnCandidateExitedAfterAdmission",
          pid: candidate.pid,
          code: 23,
          stderr: "original daemon failed",
        },
      })
    }).pipe(Effect.provide(platform))))
  })

  it("stops cleanly without an owner and never resolves or spawns", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-stop-" })
      const manager = yield* makeLocalAcnInstanceManager({
        dataDir,
        binaryPath: `${dataDir}/must-not-be-resolved`,
      }).pipe(Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
        spawn: () => Effect.dieMessage("stop must not spawn"),
      })))
      yield* manager.stop
    }).pipe(Effect.provide(platform))))
  })

  it("fails one stalled candidate at its admission deadline without respawning", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-timeout-" })
      const exact = yield* ProcessGroupControllerLive.currentProcess
      let spawns = 0
      let cleanups = 0
      const manager = yield* makeLocalAcnInstanceManager({
        dataDir,
        launchOverride: { target: SDK_ACN_TARGET, command: ["unused-test-acn"] },
      }).pipe(Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
        spawn: () => Effect.sync(() => {
          spawns += 1
        }).pipe(
          Effect.zipRight(Effect.addFinalizer(() => Effect.sync(() => {
            cleanups += 1
          }))),
          Effect.as({
            pid: exact.pid,
            exited: Effect.never,
            confirmExactProcess: () => Effect.void,
            admit: Effect.void,
            stopAndReap: Effect.void,
          }),
        ),
      })))
      const ensuring = yield* runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })).pipe(
        Effect.exit,
        Effect.fork,
      )
      while (spawns === 0) yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.seconds(31))
      yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.seconds(1))
      const result = yield* Fiber.join(ensuring)
      expect(Exit.isFailure(result)).toBe(true)
      expect(spawns).toBe(1)
      expect(cleanups).toBe(1)
    }).pipe(Effect.provide(Layer.merge(platform, TestContext.TestContext)))))
  })

  it("does not retire a live Starting owner when its activity is stable past health grace", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-starting-grace-" })
      const processFixture = yield* makeExactProcessFixture
      const exact = processFixture.exact
      const id = AcnInstanceIdSchema.make("starting-owner")
      let ready = false
      const http = makeOwnerHttp(exact, id, () => Option.some(ready
        ? { status: 200, state: new AcnReady({}) }
        : {
            status: 503,
            state: new AcnStarting({
              activity: {
                _tag: "PreparingBackend",
                backend: { _tag: "Metal", hardwareLabel: "Apple M3 Pro" },
              },
              progress: Option.none(),
            }),
          }), processFixture.stop)
      const owners = yield* makeAcnOwnerStore(dataDir)
      yield* owners.replaceOwner(Option.none(), { ...exact, port: 49152 })

      const manager = yield* makeLocalAcnInstanceManagerWithProcessController({
        dataDir,
        binaryPath: `${dataDir}/must-not-be-resolved`,
      }).pipe(
        Effect.provideService(HttpClient.HttpClient, http.client),
        Effect.provideService(ProcessGroupController, processFixture.controller),
        Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
          spawn: () => Effect.dieMessage("live starting owner must not spawn"),
        })),
      )

      const ensuring = yield* runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })).pipe(
        Effect.fork,
      )
      while (http.requests.health === 0) yield* Effect.yieldNow()
      // Beyond former HEALTH_GRACE (30s): still Starting with stable activity.
      // Under the old policy this would retire the owner and fail install.
      yield* TestClock.adjust(Duration.seconds(45))
      expect(http.requests.shutdown).toBe(0)
      ready = true
      yield* TestClock.adjust(Duration.seconds(1))
      const result = yield* Fiber.join(ensuring)
      expect(result.id).toBe(id)
    }).pipe(Effect.provide(Layer.merge(platform, TestContext.TestContext)))))
  })

  it("fails a continuously observable Starting owner at the absolute startup ceiling", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-startup-ceiling-" })
      const processFixture = yield* makeExactProcessFixture
      const exact = processFixture.exact
      const id = AcnInstanceIdSchema.make("startup-ceiling-owner")
      const http = makeOwnerHttp(exact, id, () => Option.some({
        status: 503,
        state: new AcnStarting({ activity: "Resolving", progress: Option.none() }),
      }), processFixture.stop)
      const owners = yield* makeAcnOwnerStore(dataDir)
      yield* owners.replaceOwner(Option.none(), { ...exact, port: 49152 })
      const manager = yield* makeLocalAcnInstanceManagerWithProcessController({
        dataDir,
        binaryPath: `${dataDir}/must-not-be-resolved`,
      }).pipe(
        Effect.provideService(HttpClient.HttpClient, http.client),
        Effect.provideService(ProcessGroupController, processFixture.controller),
        Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
          spawn: () => Effect.dieMessage("expired starting owner must not spawn"),
        })),
      )

      const ensuring = yield* runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })).pipe(
        Effect.either,
        Effect.fork,
      )
      while (http.requests.health === 0) yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.seconds(298))
      expect(http.requests.shutdown).toBe(0)
      yield* TestClock.adjust(Duration.seconds(3))
      while (http.requests.shutdown === 0) yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.seconds(1))
      const result = yield* Fiber.join(ensuring)
      expect(result).toMatchObject({
        _tag: "Left",
        left: { _tag: "AcnDaemonStartupTimedOut", owner: exact },
      })
    }).pipe(Effect.provide(Layer.merge(platform, TestContext.TestContext)))))
  }, 15_000)

  it("retires an owner only after health is continuously unobservable for health grace", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-health-grace-" })
      const processFixture = yield* makeExactProcessFixture
      const exact = processFixture.exact
      const id = AcnInstanceIdSchema.make("unobservable-owner")
      const http = makeOwnerHttp(exact, id, Option.none, processFixture.stop)
      const owners = yield* makeAcnOwnerStore(dataDir)
      yield* owners.replaceOwner(Option.none(), { ...exact, port: 49152 })
      const manager = yield* makeLocalAcnInstanceManagerWithProcessController({
        dataDir,
        binaryPath: `${dataDir}/must-not-be-resolved`,
      }).pipe(
        Effect.provideService(HttpClient.HttpClient, http.client),
        Effect.provideService(ProcessGroupController, processFixture.controller),
        Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
          spawn: () => Effect.dieMessage("unobservable owner must not spawn during grace"),
        })),
      )

      const ensuring = yield* runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })).pipe(Effect.fork)
      while (http.requests.health === 0) yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.seconds(29))
      expect(http.requests.shutdown).toBe(0)
      yield* TestClock.adjust(Duration.seconds(2))
      while (http.requests.shutdown === 0) yield* Effect.yieldNow()
      expect(http.requests.shutdown).toBe(1)
      yield* Fiber.interrupt(ensuring)
    }).pipe(Effect.provide(Layer.merge(platform, TestContext.TestContext)))))
  })

  it("retires an observable Stopping owner after stopping grace", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-stopping-grace-" })
      const processFixture = yield* makeExactProcessFixture
      const exact = processFixture.exact
      const id = AcnInstanceIdSchema.make("stopping-owner")
      const http = makeOwnerHttp(exact, id, () => Option.some({
        status: 503,
        state: new AcnStopping({
          reason: "administrative",
          safeDetail: Option.none(),
        }),
      }), processFixture.stop)
      const owners = yield* makeAcnOwnerStore(dataDir)
      yield* owners.replaceOwner(Option.none(), { ...exact, port: 49152 })
      const manager = yield* makeLocalAcnInstanceManagerWithProcessController({
        dataDir,
        binaryPath: `${dataDir}/must-not-be-resolved`,
      }).pipe(
        Effect.provideService(HttpClient.HttpClient, http.client),
        Effect.provideService(ProcessGroupController, processFixture.controller),
        Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
          spawn: () => Effect.dieMessage("stopping owner must not spawn during grace"),
        })),
      )

      const ensuring = yield* runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })).pipe(Effect.fork)
      while (http.requests.health === 0) yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.seconds(4))
      expect(http.requests.shutdown).toBe(0)
      yield* TestClock.adjust(Duration.seconds(2))
      while (http.requests.shutdown === 0) yield* Effect.yieldNow()
      expect(http.requests.shutdown).toBe(1)
      yield* Fiber.interrupt(ensuring)
    }).pipe(Effect.provide(Layer.merge(platform, TestContext.TestContext)))))
  })

})
