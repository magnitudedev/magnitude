import * as FileSystem from "@effect/platform/FileSystem"
import { FetchHttpClient } from "@effect/platform"
import { BunContext, BunFileSystem } from "@effect/platform-bun"
import {
  AcnInstanceIdSchema,
  AcnReady,
} from "@magnitudedev/acn-protocol"
import {
  applyAcnProcessCommand,
  readAcnProcessState,
} from "@magnitudedev/acn-protocol/process-state"
import { Deferred, Effect, Fiber, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { runAcnLaunch } from "./acn-process-manager"
import { ChildProcessSpawner, scopePreHandoffCandidate } from "./child-process"
import { makeLocalAcnProcessManager } from "./local-acn-process-manager"
import { DaemonSpawnFailed } from "./errors"

describe("LocalAcnProcessManager", () => {
  it("coalesces concurrent launches onto one admitted candidate", async () => {
    let spawns = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-manager-" })
      let assigned: {
        readonly id: string
        readonly identity: string
        readonly pid: number
      } | undefined
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: () => {
          if (assigned === undefined) return new Response(null, { status: 503 })
          return Response.json({
            service: "magnitude-acn",
            version: assigned.identity,
            id: assigned.id,
            pid: assigned.pid,
            state: new AcnReady({}),
          })
        },
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))

      const spawner = ChildProcessSpawner.of({
        spawn: () => Effect.gen(function* () {
          spawns += 1
          return yield* scopePreHandoffCandidate({
            pid: process.pid,
            stopAndReap: Effect.void,
            releaseForHandoff: Effect.gen(function* () {
              const state = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
              if (state.mode._tag !== "Changing" || state.mode.owner._tag !== "Candidate") {
                return yield* Effect.dieMessage("candidate was not published before bootstrap handoff")
              }
              const admitted = yield* applyAcnProcessCommand({
                dataDirectory,
                expectedRevision: Option.some(state.revision),
                command: {
                  _tag: "CandidateAdmitted",
                  candidate: state.mode.owner.candidate,
                  id: AcnInstanceIdSchema.make("test-acn"),
                  url: `http://127.0.0.1:${server.port}`,
                },
              })
              if (admitted.mode._tag !== "Assigned") {
                return yield* Effect.dieMessage("candidate admission did not assign the ACN")
              }
              assigned = admitted.mode.current
            }).pipe(
              Effect.provideService(FileSystem.FileSystem, fs),
              Effect.mapError((error) => new DaemonSpawnFailed({ reason: String(error) })),
            ),
          })
        }),
      })
      const manager = yield* makeLocalAcnProcessManager({ dataDir: dataDirectory }).pipe(
        Effect.provideService(ChildProcessSpawner, spawner),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const request = {
        identity: "1.0.0" as never,
        replace: Option.none(),
        command: Option.some(["fake-acn"]),
      }
      const results = yield* Effect.all([
        runAcnLaunch(manager.launch(request)),
        runAcnLaunch(manager.launch(request)),
      ], { concurrency: "unbounded" })
      expect(results.map((result) => result.id)).toEqual(["test-acn", "test-acn"])
      expect(spawns).toBe(1)
    }).pipe(Effect.provide(BunFileSystem.layer))))
  })

  it("continues an admitted change after its launch observer is canceled", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-manager-cancel-" })
      const admit = yield* Deferred.make<void>()
      let assigned: { readonly id: string; readonly identity: string; readonly pid: number } | undefined
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: () => Response.json({
          service: "magnitude-acn",
          version: assigned?.identity ?? "1.0.0",
          id: assigned?.id ?? "test-acn",
          pid: process.pid,
          state: new AcnReady({}),
        }),
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))

      const spawner = ChildProcessSpawner.of({
        spawn: () => scopePreHandoffCandidate({
          pid: process.pid,
          stopAndReap: Effect.void,
          releaseForHandoff: Effect.gen(function* () {
            yield* Deferred.await(admit)
            const state = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
            if (state.mode._tag !== "Changing" || state.mode.owner._tag !== "Candidate") {
              return yield* Effect.dieMessage("candidate was not durable at handoff")
            }
            const admitted = yield* applyAcnProcessCommand({
              dataDirectory,
              expectedRevision: Option.some(state.revision),
              command: {
                _tag: "CandidateAdmitted",
                candidate: state.mode.owner.candidate,
                id: AcnInstanceIdSchema.make("test-acn"),
                url: `http://127.0.0.1:${server.port}`,
              },
            })
            if (admitted.mode._tag !== "Assigned") return yield* Effect.dieMessage("candidate was not assigned")
            assigned = admitted.mode.current
          }).pipe(
            Effect.provideService(FileSystem.FileSystem, fs),
            Effect.mapError((error) => new DaemonSpawnFailed({ reason: String(error) })),
          ),
        }),
      })
      const manager = yield* makeLocalAcnProcessManager({ dataDir: dataDirectory }).pipe(
        Effect.provideService(ChildProcessSpawner, spawner),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const observer = yield* manager.launch({
        identity: "1.0.0" as never,
        replace: Option.none(),
        command: Option.some(["fake-acn"]),
      }).pipe(Stream.runDrain, Effect.fork)

      yield* Effect.gen(function* () {
        while (true) {
          const state = yield* readAcnProcessState(dataDirectory)
          if (Option.exists(state, (value) =>
            value.mode._tag === "Changing" && value.mode.owner._tag === "Candidate"
          )) return
          yield* Effect.sleep("10 millis")
        }
      }).pipe(Effect.timeout("2 seconds"))
      yield* Fiber.interrupt(observer)
      yield* Deferred.succeed(admit, undefined)

      const final = yield* Effect.gen(function* () {
        while (true) {
          const state = yield* readAcnProcessState(dataDirectory)
          if (Option.isSome(state) && state.value.mode._tag === "Assigned") return state.value
          yield* Effect.sleep("10 millis")
        }
      }).pipe(Effect.timeout("2 seconds"))
      expect(final.mode._tag).toBe("Assigned")
    }).pipe(Effect.provide(BunFileSystem.layer))))
  })

  it("upgrades one lower active change instead of launching beside it", async () => {
    let spawns = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-manager-upgrade-" })
      const deadManager = { pid: 999_999_998, processStartIdentity: "dead-manager" as never }
      const lower = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.none(),
        command: { _tag: "BeginEnsure", target: "1.0.0" as never, manager: deadManager },
      })
      if (lower.mode._tag !== "Changing") return yield* Effect.dieMessage("ensure did not begin")
      const candidate = {
        identity: "1.0.0" as never,
        pid: 999_999_999,
        processStartIdentity: "dead-candidate" as never,
      }
      yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(lower.revision),
        command: { _tag: "CandidateSpawned", manager: deadManager, candidate },
      })

      let assigned: { readonly id: string; readonly identity: string; readonly pid: number } | undefined
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: () => Response.json({
          service: "magnitude-acn",
          version: assigned?.identity ?? "2.0.0",
          id: assigned?.id ?? "upgraded-acn",
          pid: process.pid,
          state: new AcnReady({}),
        }),
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))
      const spawner = ChildProcessSpawner.of({
        spawn: () => Effect.gen(function* () {
          spawns += 1
          return yield* scopePreHandoffCandidate({
            pid: process.pid,
            stopAndReap: Effect.void,
            releaseForHandoff: Effect.gen(function* () {
              const state = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
              if (state.mode._tag !== "Changing" || state.mode.owner._tag !== "Candidate") {
                return yield* Effect.dieMessage("upgraded candidate was not published")
              }
              const admitted = yield* applyAcnProcessCommand({
                dataDirectory,
                expectedRevision: Option.some(state.revision),
                command: {
                  _tag: "CandidateAdmitted",
                  candidate: state.mode.owner.candidate,
                  id: AcnInstanceIdSchema.make("upgraded-acn"),
                  url: `http://127.0.0.1:${server.port}`,
                },
              })
              if (admitted.mode._tag !== "Assigned") return yield* Effect.dieMessage("upgrade was not assigned")
              assigned = admitted.mode.current
            }).pipe(
              Effect.provideService(FileSystem.FileSystem, fs),
              Effect.mapError((error) => new DaemonSpawnFailed({ reason: String(error) })),
            ),
          })
        }),
      })
      const manager = yield* makeLocalAcnProcessManager({ dataDir: dataDirectory }).pipe(
        Effect.provideService(ChildProcessSpawner, spawner),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const result = yield* runAcnLaunch(manager.launch({
        identity: "2.0.0" as never,
        replace: Option.none(),
        command: Option.some(["fake-acn-v2"]),
      }))
      const final = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
      expect(result.identity).toBe("2.0.0")
      expect(final.identityFloor).toBe("2.0.0")
      expect(final.mode._tag).toBe("Assigned")
      if (final.mode._tag !== "Assigned") return
      expect(Option.getOrThrow(final.mode.result)).toMatchObject({
        _tag: "Admitted",
        changeRevision: lower.mode.changeRevision,
      })
      expect(spawns).toBe(1)
    }).pipe(Effect.provide(BunFileSystem.layer))))
  })
})
