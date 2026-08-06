import * as FileSystem from "@effect/platform/FileSystem"
import { FetchHttpClient } from "@effect/platform"
import { BunContext, BunFileSystem } from "@effect/platform-bun"
import {
  AcnInstanceIdSchema,
  AcnReady,
} from "@magnitudedev/acn-protocol"
import {
  applyAcnProcessCommand,
  currentProcessStartIdentity,
  readAcnProcessState,
  readProcessStartIdentity,
  type AssignedAcn,
} from "@magnitudedev/acn-protocol/process-state"
import { Deferred, Effect, Fiber, Layer, Option, Runtime, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { runAcnEnsure } from "./acn-ensurer"
import { ChildProcessSpawner, scopePreHandoffCandidate } from "./child-process"
import {
  AcnLaunchSource,
  makeLocalAcnDaemonAdministrator,
  makeLocalAcnEnsurer,
  resolveAssignedAcnProxyTarget,
} from "./local-acn-ensurer"
import { AcnEnsuranceFailed, DownloadFailed } from "./errors"

describe("LocalAcnEnsurer", () => {
  it("resolves proxy targets only from the exact stable assignment", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-proxy-target-" })
      const manager = {
        pid: process.pid,
        processStartIdentity: yield* currentProcessStartIdentity,
      }
      const begun = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.none(),
        command: { _tag: "BeginEnsure", target: "1.0.0" as never, manager },
      })
      const prepared = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(begun.revision),
        command: { _tag: "PreparationSucceeded", manager },
      })
      const candidate = { identity: "1.0.0" as never, ...manager }
      const spawned = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(prepared.revision),
        command: { _tag: "CandidateSpawned", manager, candidate },
      })
      const id = AcnInstanceIdSchema.make("proxy-acn")
      const assigned = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(spawned.revision),
        command: {
          _tag: "CandidateAdmitted",
          candidate,
          id,
          url: "http://127.0.0.1:5000",
        },
      })
      expect(yield* resolveAssignedAcnProxyTarget(dataDirectory, id)).toEqual(
        Option.some("http://127.0.0.1:5000"),
      )
      if (assigned.mode._tag !== "Assigned") return yield* Effect.dieMessage("ACN was not assigned")
      yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(assigned.revision),
        command: {
          _tag: "BeginReplacement",
          target: assigned.identityFloor,
          manager,
          current: assigned.mode.current,
        },
      })
      expect(Option.isNone(yield* resolveAssignedAcnProxyTarget(dataDirectory, id))).toBe(true)
    }).pipe(Effect.provide(BunContext.layer))))
  })

  it("accepts readiness when the final reread changes only ICN ownership revision", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-ready-reread-" })
      const manager = {
        pid: process.pid,
        processStartIdentity: yield* currentProcessStartIdentity,
      }
      const runPromise = Runtime.runPromise(yield* Effect.runtime<FileSystem.FileSystem>())
      let assignedAcn: AssignedAcn | undefined
      let recorded = false
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: async () => {
          const current = assignedAcn
          if (current === undefined) return new Response(null, { status: 503 })
          if (!recorded) {
            recorded = true
            const state = Option.getOrThrow(await runPromise(readAcnProcessState(dataDirectory)))
            await runPromise(applyAcnProcessCommand({
              dataDirectory,
              expectedRevision: Option.some(state.revision),
              command: {
                _tag: "RecordIcn",
                acn: current,
                icn: { id: "icn-1", ...manager },
              },
            }))
          }
          return Response.json({
            service: "magnitude-acn",
            version: current.identity,
            id: current.id,
            pid: current.pid,
            state: new AcnReady({}),
          })
        },
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))
      const begun = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.none(),
        command: { _tag: "BeginEnsure", target: "1.0.0" as never, manager },
      })
      const prepared = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(begun.revision),
        command: { _tag: "PreparationSucceeded", manager },
      })
      const candidate = { identity: "1.0.0" as never, ...manager }
      const spawned = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(prepared.revision),
        command: { _tag: "CandidateSpawned", manager, candidate },
      })
      const assigned = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(spawned.revision),
        command: {
          _tag: "CandidateAdmitted",
          candidate,
          id: AcnInstanceIdSchema.make("ready-reread-acn"),
          url: `http://127.0.0.1:${server.port}`,
        },
      })
      if (assigned.mode._tag !== "Assigned") return yield* Effect.dieMessage("ACN was not assigned")
      assignedAcn = assigned.mode.current
      const ensurer = yield* makeLocalAcnEnsurer({ dataDir: dataDirectory }).pipe(
        Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
          spawn: () => Effect.dieMessage("stable ready assignment must not spawn"),
        })),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const result = yield* runAcnEnsure(ensurer.ensure({ minimumIdentity: "1.0.0" as never }))
      expect(result.id).toBe(assignedAcn.id)
      const final = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
      expect(final.mode._tag === "Assigned" && Option.isSome(final.mode.current.ownedIcn)).toBe(true)
    }).pipe(Effect.provide(BunContext.layer))))
  })

  it("administrative stop is a no-op without durable assignment state", async () => {
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const fs = yield* FileSystem.FileSystem
          const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-stop-empty-" })
          const administrator = yield* makeLocalAcnDaemonAdministrator({ dataDir: dataDirectory }).pipe(
            Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer))
          )
          yield* administrator.stopCurrent
          expect(Option.isNone(yield* readAcnProcessState(dataDirectory))).toBe(true)
        })
      ).pipe(Effect.provide(BunFileSystem.layer))
    )
  })

  it("does not retire the incumbent when replacement preparation fails", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-prepare-failure-" })
      const allowFailure = yield* Deferred.make<void>()
      const child = Bun.spawn([process.execPath, "-e", "setInterval(() => {}, 1000)"])
      yield* Effect.addFinalizer(() => Effect.sync(() => child.kill()))
      const manager = {
        pid: process.pid,
        processStartIdentity: yield* currentProcessStartIdentity,
      }
      const candidate = {
        identity: "1.0.0" as never,
        pid: child.pid,
        processStartIdentity: Option.getOrThrow(yield* readProcessStartIdentity(child.pid)),
      }
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: () => Response.json({
          service: "magnitude-acn",
          version: candidate.identity,
          id: "incumbent-acn",
          pid: candidate.pid,
          state: new AcnReady({}),
        }),
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))
      const begun = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.none(),
        command: { _tag: "BeginEnsure", target: candidate.identity, manager },
      })
      const prepared = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(begun.revision),
        command: { _tag: "PreparationSucceeded", manager },
      })
      const spawned = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(prepared.revision),
        command: { _tag: "CandidateSpawned", manager, candidate },
      })
      const assigned = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(spawned.revision),
        command: {
          _tag: "CandidateAdmitted",
          candidate,
          id: AcnInstanceIdSchema.make("incumbent-acn"),
          url: `http://127.0.0.1:${server.port}`,
        },
      })
      const launchSource = AcnLaunchSource.of({
          supports: (identity) => identity === "2.0.0",
          prepare: () => Deferred.await(allowFailure).pipe(Effect.zipRight(
            Effect.fail(new DownloadFailed({
              url: "https://example.invalid/acn",
              status: 503,
              reason: "unavailable",
            })),
          )),
      })
      const ensurer = yield* makeLocalAcnEnsurer({ dataDir: dataDirectory }).pipe(
        Effect.provideService(AcnLaunchSource, launchSource),
        Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
          spawn: () => Effect.dieMessage("failed preparation must not spawn"),
        })),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const replacement = yield* runAcnEnsure(ensurer.ensure({ minimumIdentity: "2.0.0" as never })).pipe(
        Effect.either,
        Effect.fork,
      )
      yield* Effect.gen(function* () {
        while (true) {
          const state = yield* readAcnProcessState(dataDirectory)
          if (Option.exists(state, (value) =>
            value.mode._tag === "Changing" &&
            value.mode.owner._tag === "Manager" &&
            value.mode.owner.phase._tag === "Preparing"
          )) return
          yield* Effect.sleep("10 millis")
        }
      }).pipe(Effect.timeout("2 seconds"))
      const follower = yield* runAcnEnsure(ensurer.ensure({ minimumIdentity: "1.0.0" as never })).pipe(Effect.fork)
      yield* Effect.sleep("20 millis")
      yield* Deferred.succeed(allowFailure, undefined)
      const result = yield* Fiber.join(replacement)
      const retained = yield* Fiber.join(follower)
      expect(result._tag).toBe("Left")
      const final = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
      expect(final.mode._tag).toBe("Assigned")
      if (assigned.mode._tag !== "Assigned" || final.mode._tag !== "Assigned") return
      expect(final.mode.current).toEqual(assigned.mode.current)
      expect(Option.isSome(yield* readProcessStartIdentity(child.pid))).toBe(true)
      expect(retained.id).toBe("incumbent-acn")
    }).pipe(Effect.provide(BunContext.layer))))
  })

  it("administrative stop cancels an unpublished spawning intent", async () => {
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const fs = yield* FileSystem.FileSystem
          const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-stop-spawning-" })
          yield* applyAcnProcessCommand({
            dataDirectory,
            expectedRevision: Option.none(),
            command: {
              _tag: "BeginEnsure",
              target: "1.0.0" as never,
              manager: {
                pid: process.pid,
                processStartIdentity: "superseded-manager" as never,
              },
            },
          })
          const administrator = yield* makeLocalAcnDaemonAdministrator({ dataDir: dataDirectory }).pipe(
            Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer))
          )
          yield* administrator.stopCurrent
          const stopped = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
          expect(stopped.mode._tag).toBe("Unassigned")
          if (stopped.mode._tag !== "Unassigned") return
          expect(Option.getOrThrow(stopped.mode.result)._tag).toBe("Terminated")
        })
      ).pipe(Effect.provide(BunFileSystem.layer))
    )
  })

  it("administrative stop gracefully retires the exact assigned ACN", async () => {
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const fs = yield* FileSystem.FileSystem
          const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-stop-assigned-" })
          const child = Bun.spawn([
            process.execPath,
            "-e",
            "setInterval(() => {}, 1000)",
          ])
          yield* Effect.addFinalizer(() => Effect.sync(() => child.kill()))
          const processStartIdentity = Option.getOrThrow(
            yield* readProcessStartIdentity(child.pid),
          )
          const manager = {
            pid: process.pid,
            processStartIdentity: yield* currentProcessStartIdentity,
          }
          const server = Bun.serve({
            port: 0,
            hostname: "127.0.0.1",
            fetch: (request) => {
              if (new URL(request.url).pathname === "/shutdown") {
                child.kill("SIGTERM")
                return new Response(null, { status: 204 })
              }
              return new Response(null, { status: 404 })
            },
          })
          yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))

          const begun = yield* applyAcnProcessCommand({
            dataDirectory,
            expectedRevision: Option.none(),
            command: { _tag: "BeginEnsure", target: "1.0.0" as never, manager },
          })
          const prepared = yield* applyAcnProcessCommand({
            dataDirectory,
            expectedRevision: Option.some(begun.revision),
            command: { _tag: "PreparationSucceeded", manager },
          })
          const candidate = {
            identity: "1.0.0" as never,
            pid: child.pid,
            processStartIdentity,
          }
          const spawned = yield* applyAcnProcessCommand({
            dataDirectory,
            expectedRevision: Option.some(prepared.revision),
            command: { _tag: "CandidateSpawned", manager, candidate },
          })
          yield* applyAcnProcessCommand({
            dataDirectory,
            expectedRevision: Option.some(spawned.revision),
            command: {
              _tag: "CandidateAdmitted",
              candidate,
              id: AcnInstanceIdSchema.make("assigned-acn"),
              url: `http://127.0.0.1:${server.port}`,
            },
          })

          const administrator = yield* makeLocalAcnDaemonAdministrator({ dataDir: dataDirectory }).pipe(
            Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer))
          )
          yield* administrator.stopCurrent

          const stopped = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
          expect(stopped.mode._tag).toBe("Unassigned")
          if (stopped.mode._tag !== "Unassigned") return
          expect(Option.getOrThrow(stopped.mode.result)._tag).toBe("Terminated")
          expect(Option.isNone(yield* readProcessStartIdentity(child.pid))).toBe(true)
        })
      ).pipe(Effect.provide(BunContext.layer))
    )
  })

  it("coalesces concurrent ensures onto one admitted candidate", async () => {
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
              Effect.mapError((error) => new AcnEnsuranceFailed({ reason: String(error) })),
            ),
          })
        }),
      })
      const ensurer = yield* makeLocalAcnEnsurer({
        dataDir: dataDirectory,
        launchOverride: { identity: "1.0.0" as never, command: ["fake-acn"] },
      }).pipe(
        Effect.provideService(ChildProcessSpawner, spawner),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const request = { minimumIdentity: "1.0.0" as never }
      const results = yield* Effect.all([
        runAcnEnsure(ensurer.ensure(request)),
        runAcnEnsure(ensurer.ensure(request)),
      ], { concurrency: "unbounded" })
      expect(results.map((result) => result.id)).toEqual(["test-acn", "test-acn"])
      expect(spawns).toBe(1)
    }).pipe(Effect.provide(BunFileSystem.layer))))
  })

  it("serializes one local supervisor when a preparing target is upgraded", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-preparing-upgrade-" })
      const lowerPreparationStarted = yield* Deferred.make<void>()
      const releaseLowerPreparation = yield* Deferred.make<void>()
      const preparations: string[] = []
      let assigned: AssignedAcn | undefined
      let spawns = 0
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: () => assigned === undefined
          ? new Response(null, { status: 503 })
          : Response.json({
              service: "magnitude-acn",
              version: assigned.identity,
              id: assigned.id,
              pid: assigned.pid,
              state: new AcnReady({}),
            }),
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))
      const launchSource = AcnLaunchSource.of({
        supports: () => true,
        prepare: (identity) => Effect.gen(function* () {
          preparations.push(identity)
          if (identity === "1.0.0") {
            yield* Deferred.succeed(lowerPreparationStarted, undefined)
            yield* Deferred.await(releaseLowerPreparation)
          }
          return { identity, command: [identity] as const }
        }),
      })
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
                  id: AcnInstanceIdSchema.make("upgraded-preparation-acn"),
                  url: `http://127.0.0.1:${server.port}`,
                },
              })
              if (admitted.mode._tag !== "Assigned") return yield* Effect.dieMessage("upgrade was not assigned")
              assigned = admitted.mode.current
            }).pipe(
              Effect.provideService(FileSystem.FileSystem, fs),
              Effect.mapError((error) => new AcnEnsuranceFailed({ reason: String(error) })),
            ),
          })
        }),
      })
      const ensurer = yield* makeLocalAcnEnsurer({ dataDir: dataDirectory }).pipe(
        Effect.provideService(AcnLaunchSource, launchSource),
        Effect.provideService(ChildProcessSpawner, spawner),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const lower = yield* runAcnEnsure(ensurer.ensure({ minimumIdentity: "1.0.0" as never })).pipe(Effect.fork)
      yield* Deferred.await(lowerPreparationStarted)
      const higher = yield* runAcnEnsure(ensurer.ensure({ minimumIdentity: "2.0.0" as never })).pipe(Effect.fork)
      yield* Effect.gen(function* () {
        while (true) {
          const state = yield* readAcnProcessState(dataDirectory)
          if (Option.exists(state, (value) =>
            value.mode._tag === "Changing" &&
            value.mode.purpose._tag === "Ensure" &&
            value.mode.purpose.target === "2.0.0"
          )) return
          yield* Effect.sleep("10 millis")
        }
      }).pipe(Effect.timeout("2 seconds"))
      yield* Deferred.succeed(releaseLowerPreparation, undefined)
      const [lowerResult, higherResult] = yield* Effect.all([
        Fiber.join(lower),
        Fiber.join(higher),
      ], { concurrency: "unbounded" })
      expect(lowerResult.identity).toBe("2.0.0")
      expect(higherResult.identity).toBe("2.0.0")
      expect(preparations).toEqual(["1.0.0", "2.0.0"])
      expect(spawns).toBe(1)
    }).pipe(Effect.provide(BunFileSystem.layer))))
  })

  it("lets an older development client follow a newer owner's preparation", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-dev-follower-" })
      const allowSpawn = yield* Deferred.make<void>()
      let assigned: AssignedAcn | undefined
      let spawns = 0
      const olderIdentity = "1.0.0+dev.test.1" as never
      const newerIdentity = "1.0.0+dev.test.2" as never
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: () => assigned === undefined
          ? new Response(null, { status: 503 })
          : Response.json({
              service: "magnitude-acn",
              version: assigned.identity,
              id: assigned.id,
              pid: assigned.pid,
              state: new AcnReady({}),
            }),
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))

      const spawner = ChildProcessSpawner.of({
        spawn: () => Effect.gen(function* () {
          spawns += 1
          yield* Deferred.await(allowSpawn)
          return yield* scopePreHandoffCandidate({
            pid: process.pid,
            stopAndReap: Effect.void,
            releaseForHandoff: Effect.gen(function* () {
              const state = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
              if (state.mode._tag !== "Changing" || state.mode.owner._tag !== "Candidate") {
                return yield* Effect.dieMessage("development candidate was not published")
              }
              const admitted = yield* applyAcnProcessCommand({
                dataDirectory,
                expectedRevision: Option.some(state.revision),
                command: {
                  _tag: "CandidateAdmitted",
                  candidate: state.mode.owner.candidate,
                  id: AcnInstanceIdSchema.make("newer-dev-acn"),
                  url: `http://127.0.0.1:${server.port}`,
                },
              })
              if (admitted.mode._tag !== "Assigned") return yield* Effect.dieMessage("development ACN was not assigned")
              assigned = admitted.mode.current
            }).pipe(
              Effect.provideService(FileSystem.FileSystem, fs),
              Effect.mapError((error) => new AcnEnsuranceFailed({ reason: String(error) })),
            ),
          })
        }),
      })
      const makeEnsurer = (identity: never, command: string) => makeLocalAcnEnsurer({
        dataDir: dataDirectory,
        launchOverride: { identity, command: [command] },
      }).pipe(
        Effect.provideService(ChildProcessSpawner, spawner),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const newer = yield* makeEnsurer(newerIdentity, "newer-dev-acn")
      const older = yield* makeEnsurer(olderIdentity, "older-dev-acn")
      const newerFiber = yield* runAcnEnsure(newer.ensure({ minimumIdentity: newerIdentity })).pipe(Effect.fork)
      yield* Effect.gen(function* () {
        while (true) {
          const state = yield* readAcnProcessState(dataDirectory)
          if (Option.exists(state, (value) =>
            value.mode._tag === "Changing" &&
            value.mode.purpose._tag === "Ensure" &&
            value.mode.purpose.target === newerIdentity
          )) return
          yield* Effect.sleep("10 millis")
        }
      }).pipe(Effect.timeout("2 seconds"))
      const olderFiber = yield* runAcnEnsure(older.ensure({ minimumIdentity: olderIdentity })).pipe(Effect.fork)
      yield* Deferred.succeed(allowSpawn, undefined)
      const [newerResult, olderResult] = yield* Effect.all([
        Fiber.join(newerFiber),
        Fiber.join(olderFiber),
      ], { concurrency: "unbounded" })
      expect(newerResult.id).toBe("newer-dev-acn")
      expect(olderResult.id).toBe("newer-dev-acn")
      expect(spawns).toBe(1)
    }).pipe(Effect.provide(BunFileSystem.layer))))
  })

  it("keeps an observer bound to its exact change when a later change begins", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-exact-change-" })
      const manager = {
        pid: process.pid,
        processStartIdentity: yield* currentProcessStartIdentity,
      }
      const first = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.none(),
        command: { _tag: "BeginEnsure", target: "1.0.0" as never, manager },
      })
      const ensurer = yield* makeLocalAcnEnsurer({
        dataDir: dataDirectory,
        launchOverride: { identity: "1.0.0" as never, command: ["fake-acn"] },
      }).pipe(
        Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
          spawn: () => Effect.dieMessage("a follower must not spawn"),
        })),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const observer = yield* runAcnEnsure(ensurer.ensure({ minimumIdentity: "1.0.0" as never })).pipe(
        Effect.either,
        Effect.fork,
      )
      yield* Effect.sleep("20 millis")
      const failed = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(first.revision),
        command: { _tag: "PreparationFailed", manager, reason: "first change failed" },
      })
      yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(failed.revision),
        command: { _tag: "BeginEnsure", target: "1.0.0" as never, manager },
      })
      const result = yield* Fiber.join(observer).pipe(Effect.timeout("2 seconds"))
      expect(result._tag).toBe("Left")
      if (result._tag === "Right") return
      expect(result.left).toBeInstanceOf(AcnEnsuranceFailed)
      if (!(result.left instanceof AcnEnsuranceFailed)) return
      expect(result.left.reason).toBe("first change failed")
    }).pipe(Effect.provide(BunContext.layer))))
  })

  it("explicitly follows a successful change superseded by a sufficient replacement", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-successor-change-" })
      const manager = {
        pid: process.pid,
        processStartIdentity: yield* currentProcessStartIdentity,
      }
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: () => Response.json({
          service: "magnitude-acn",
          version: "1.0.0",
          id: "first-acn",
          pid: process.pid,
          state: new AcnReady({}),
        }),
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => server.stop(true)))
      const first = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.none(),
        command: { _tag: "BeginEnsure", target: "1.0.0" as never, manager },
      })
      const ensurer = yield* makeLocalAcnEnsurer({
        dataDir: dataDirectory,
        launchOverride: { identity: "1.0.0" as never, command: ["first-acn"] },
      }).pipe(
        Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
          spawn: () => Effect.dieMessage("a follower must not spawn"),
        })),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const observer = yield* runAcnEnsure(ensurer.ensure({ minimumIdentity: "1.0.0" as never })).pipe(Effect.fork)
      yield* Effect.sleep("20 millis")
      const prepared = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(first.revision),
        command: { _tag: "PreparationSucceeded", manager },
      })
      const candidate = { identity: "1.0.0" as never, ...manager }
      const spawned = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(prepared.revision),
        command: { _tag: "CandidateSpawned", manager, candidate },
      })
      const admitted = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(spawned.revision),
        command: {
          _tag: "CandidateAdmitted",
          candidate,
          id: AcnInstanceIdSchema.make("first-acn"),
          url: `http://127.0.0.1:${server.port}`,
        },
      })
      if (admitted.mode._tag !== "Assigned") return yield* Effect.dieMessage("first ACN was not assigned")
      const successor = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(admitted.revision),
        command: {
          _tag: "BeginReplacement",
          target: "2.0.0" as never,
          manager,
          current: admitted.mode.current,
        },
      })
      yield* Effect.sleep("20 millis")
      yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(successor.revision),
        command: { _tag: "PreparationFailed", manager, reason: "successor unavailable" },
      })
      const result = yield* Fiber.join(observer).pipe(Effect.timeout("2 seconds"))
      expect(result.id).toBe("first-acn")
    }).pipe(Effect.provide(BunContext.layer))))
  })

  it("continues an admitted change after its ensure observer is canceled", async () => {
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
            Effect.mapError((error) => new AcnEnsuranceFailed({ reason: String(error) })),
          ),
        }),
      })
      const ensurer = yield* makeLocalAcnEnsurer({
        dataDir: dataDirectory,
        launchOverride: { identity: "1.0.0" as never, command: ["fake-acn"] },
      }).pipe(
        Effect.provideService(ChildProcessSpawner, spawner),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const observer = yield* ensurer.ensure({ minimumIdentity: "1.0.0" as never }).pipe(
        Stream.runDrain,
        Effect.fork,
      )

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

  it("waits for an admitted lower candidate before starting a higher change", async () => {
    let spawns = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-manager-upgrade-" })
      const deadManager = { pid: process.pid, processStartIdentity: "stale-manager" as never }
      const lower = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.none(),
        command: { _tag: "BeginEnsure", target: "1.0.0" as never, manager: deadManager },
      })
      const lowerPrepared = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(lower.revision),
        command: { _tag: "PreparationSucceeded", manager: deadManager },
      })
      if (lower.mode._tag !== "Changing") return yield* Effect.dieMessage("ensure did not begin")
      const candidate = {
        identity: "1.0.0" as never,
        pid: process.pid,
        processStartIdentity: "stale-candidate" as never,
      }
      yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(lowerPrepared.revision),
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
              Effect.mapError((error) => new AcnEnsuranceFailed({ reason: String(error) })),
            ),
          })
        }),
      })
      const ensurer = yield* makeLocalAcnEnsurer({
        dataDir: dataDirectory,
        launchOverride: { identity: "2.0.0" as never, command: ["fake-acn-v2"] },
      }).pipe(
        Effect.provideService(ChildProcessSpawner, spawner),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const result = yield* runAcnEnsure(ensurer.ensure({ minimumIdentity: "2.0.0" as never }))
      const final = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
      expect(result.identity).toBe("2.0.0")
      expect(final.identityFloor).toBe("2.0.0")
      expect(final.mode._tag).toBe("Assigned")
      if (final.mode._tag !== "Assigned") return
      expect(Option.getOrThrow(final.mode.result)).toMatchObject({
        _tag: "Admitted",
      })
      expect(Option.getOrThrow(final.mode.result).changeRevision).not.toBe(lower.mode.changeRevision)
      expect(spawns).toBe(1)
    }).pipe(Effect.provide(BunFileSystem.layer))))
  })

  it("joins RetiringAssigned without health-probing the retiring predecessor", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDirectory = yield* fs.makeTempDirectoryScoped({ prefix: "acn-retiring-join-" })
      const predecessor = Bun.spawn([process.execPath, "-e", "setInterval(() => {}, 1000)"])
      yield* Effect.addFinalizer(() => Effect.sync(() => predecessor.kill()))
      const predecessorStart = Option.getOrThrow(yield* readProcessStartIdentity(predecessor.pid))
      let predecessorHealthReads = 0
      const predecessorServer = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: (request) => {
          if (new URL(request.url).pathname === "/shutdown") {
            predecessor.kill("SIGTERM")
            return new Response(null, { status: 202 })
          }
          predecessorHealthReads += 1
          return new Response("retiring", { status: 503 })
        },
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => predecessorServer.stop(true)))

      const originalManager = {
        pid: process.pid,
        processStartIdentity: yield* currentProcessStartIdentity,
      }
      const begun = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.none(),
        command: { _tag: "BeginEnsure", target: "1.0.0" as never, manager: originalManager },
      })
      const prepared = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(begun.revision),
        command: { _tag: "PreparationSucceeded", manager: originalManager },
      })
      const predecessorCandidate = {
        identity: "1.0.0" as never,
        pid: predecessor.pid,
        processStartIdentity: predecessorStart,
      }
      const spawned = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(prepared.revision),
        command: { _tag: "CandidateSpawned", manager: originalManager, candidate: predecessorCandidate },
      })
      const assigned = yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(spawned.revision),
        command: {
          _tag: "CandidateAdmitted",
          candidate: predecessorCandidate,
          id: AcnInstanceIdSchema.make("retiring-acn"),
          url: `http://127.0.0.1:${predecessorServer.port}`,
        },
      })
      if (assigned.mode._tag !== "Assigned") return yield* Effect.dieMessage("predecessor was not assigned")
      yield* applyAcnProcessCommand({
        dataDirectory,
        expectedRevision: Option.some(assigned.revision),
        command: {
          _tag: "BeginReplacement",
          target: "1.0.0" as never,
          manager: { pid: process.pid, processStartIdentity: "stale-manager" as never },
          current: assigned.mode.current,
        },
      })

      let successor: { readonly id: string; readonly identity: string; readonly pid: number } | undefined
      const successorServer = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch: () => Response.json({
          service: "magnitude-acn",
          version: successor?.identity ?? "1.0.0",
          id: successor?.id ?? "successor-acn",
          pid: process.pid,
          state: new AcnReady({}),
        }),
      })
      yield* Effect.addFinalizer(() => Effect.sync(() => successorServer.stop(true)))
      const spawner = ChildProcessSpawner.of({
        spawn: () => scopePreHandoffCandidate({
          pid: process.pid,
          stopAndReap: Effect.void,
          releaseForHandoff: Effect.gen(function* () {
            const state = Option.getOrThrow(yield* readAcnProcessState(dataDirectory))
            if (state.mode._tag !== "Changing" || state.mode.owner._tag !== "Candidate") {
              return yield* Effect.dieMessage("successor candidate was not published")
            }
            const admitted = yield* applyAcnProcessCommand({
              dataDirectory,
              expectedRevision: Option.some(state.revision),
              command: {
                _tag: "CandidateAdmitted",
                candidate: state.mode.owner.candidate,
                id: AcnInstanceIdSchema.make("successor-acn"),
                url: `http://127.0.0.1:${successorServer.port}`,
              },
            })
            if (admitted.mode._tag !== "Assigned") return yield* Effect.dieMessage("successor was not assigned")
            successor = admitted.mode.current
          }).pipe(
            Effect.provideService(FileSystem.FileSystem, fs),
            Effect.mapError((error) => new AcnEnsuranceFailed({ reason: String(error) })),
          ),
        }),
      })
      const ensurer = yield* makeLocalAcnEnsurer({
        dataDir: dataDirectory,
        launchOverride: { identity: "1.0.0" as never, command: ["fake-acn"] },
      }).pipe(
        Effect.provideService(ChildProcessSpawner, spawner),
        Effect.provide(Layer.mergeAll(BunContext.layer, FetchHttpClient.layer)),
      )
      const result = yield* runAcnEnsure(ensurer.ensure({ minimumIdentity: "1.0.0" as never }))
      expect(result.id).toBe("successor-acn")
      expect(predecessorHealthReads).toBe(0)
    }).pipe(Effect.provide(BunContext.layer))))
  })
})
