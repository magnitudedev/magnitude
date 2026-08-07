import * as FileSystem from "@effect/platform/FileSystem"
import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import {
  AcnIdentitySchema,
  AcnInstanceIdSchema,
  AcnReady,
  AcnRevisionSchema,
  type AcnTarget,
} from "@magnitudedev/acn-protocol"
import {
  ExactProcessControllerLive,
  makeAcnOwnerStore,
  makeAcnRevisionStore,
} from "@magnitudedev/acn-protocol/coordination"
import { Duration, Effect, Exit, Fiber, Layer, Option, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { runAcnEnsure } from "./acn-instance-manager"
import { ChildProcessSpawner } from "./child-process"
import { AcnEnsuranceFailed } from "./errors"
import { makeLocalAcnInstanceManager } from "./local-acn-instance-manager"
import { BunSqliteDriverLayer } from "@magnitudedev/acn-protocol/coordination/bun"
import { SDK_ACN_TARGET, SDK_VERSION } from "../version"

const platform = Layer.mergeAll(BunContext.layer, FetchHttpClient.layer, BunSqliteDriverLayer)

describe("LocalAcnInstanceManager", () => {
  it("adopts a newer selected owner without resolving or spawning the caller's older artifact", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-adopt-" })
      const requested: AcnTarget = {
        revision: AcnRevisionSchema.make(1_000_000),
        identity: AcnIdentitySchema.make("1.0.0"),
      }
      const selected = AcnRevisionSchema.make(2_000_000)
      const identity = AcnIdentitySchema.make("2.0.0")
      const exact = yield* ExactProcessControllerLive.current
      const store = yield* makeAcnRevisionStore(dataDir)
      yield* store.register(selected)
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
      yield* owners.replaceOwner(Option.none(), { ...exact, port: server.port }, selected)

      const manager = yield* makeLocalAcnInstanceManager({
        dataDir,
        binaryPath: `${dataDir}/must-not-be-resolved`,
      }).pipe(Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
        spawn: () => Effect.dieMessage("a valid selected owner must not spawn"),
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
      const exact = yield* ExactProcessControllerLive.current
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
      const revisions = yield* makeAcnRevisionStore(dataDir)
      const owners = yield* makeAcnOwnerStore(dataDir)
      yield* revisions.register(selected)
      yield* owners.replaceOwner(Option.none(), { ...exact, port: server.port }, selected)
      const manager = yield* makeLocalAcnInstanceManager({
        dataDir,
        binaryPath: `${dataDir}/missing-acn`,
      }).pipe(Effect.provideService(ChildProcessSpawner, ChildProcessSpawner.of({
        spawn: () => Effect.dieMessage("selection preparation must fail before spawning"),
      })))

      expect(Exit.isFailure(yield* Effect.exit(
        runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })),
      ))).toBe(true)
    }).pipe(Effect.provide(platform))))
  })

  it("prepares and hands off one candidate only when the selected target has no owner", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const dataDir = yield* fs.makeTempDirectoryScoped({ prefix: "acn-instance-manager-launch-" })
      const exact = yield* ExactProcessControllerLive.current
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
      let spawns = 0
      const spawner = ChildProcessSpawner.of({
        spawn: () => Effect.gen(function* () {
          spawns += 1
          const expected = yield* owners.current
          const replaced = yield* owners.replaceOwner(
            expected,
            { ...exact, port: server.port! },
            SDK_ACN_TARGET.revision,
          )
          if (replaced._tag !== "Replaced") return yield* Effect.dieMessage("candidate was not admitted")
          return {
            pid: exact.pid,
            exited: Effect.never,
            admit: Effect.void,
          }
        }).pipe(Effect.mapError((error) => new AcnEnsuranceFailed({ reason: String(error) }))),
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
      const exact = yield* ExactProcessControllerLive.current
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
          Effect.as({ pid: exact.pid, exited: Effect.never, admit: Effect.void }),
        ),
      })))
      const ensuring = yield* runAcnEnsure(manager.ensure({ target: SDK_ACN_TARGET })).pipe(
        Effect.exit,
        Effect.fork,
      )
      while (spawns === 0) yield* Effect.yieldNow()
      yield* TestClock.adjust(Duration.seconds(31))
      const result = yield* Fiber.join(ensuring)
      expect(Exit.isFailure(result)).toBe(true)
      expect(spawns).toBe(1)
      expect(cleanups).toBe(1)
    }).pipe(Effect.provide(Layer.merge(platform, TestContext.TestContext)))))
  })

})
