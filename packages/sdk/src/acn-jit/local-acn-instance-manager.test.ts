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
  makeAcnOwnerLock,
  makeAcnRevisionStore,
} from "@magnitudedev/acn-protocol/coordination"
import { Effect, Exit, Layer, Option, Scope } from "effect"
import { describe, expect, it } from "vitest"
import { runAcnEnsure } from "./acn-instance-manager"
import { ChildProcessSpawner } from "./child-process"
import { AcnEnsuranceFailed } from "./errors"
import { makeLocalAcnInstanceManager } from "./local-acn-instance-manager"
import { BunSqliteMutexLayer } from "@magnitudedev/acn-protocol/coordination/bun"
import { SDK_ACN_TARGET, SDK_VERSION } from "../version"

const platform = Layer.mergeAll(BunContext.layer, FetchHttpClient.layer, BunSqliteMutexLayer)

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
      yield* store.registerPublished(selected)
      const ownerLock = yield* makeAcnOwnerLock(dataDir)
      const owner = Option.getOrThrow(yield* ownerLock.tryAcquire)
      yield* Effect.addFinalizer(() => owner.close)
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
      yield* owner.publish({ ...exact, port: server.port })

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
      const ownerLock = yield* makeAcnOwnerLock(dataDir)
      let spawns = 0
      const candidateScope = yield* Scope.make()
      yield* Effect.addFinalizer(() => Scope.close(candidateScope, Exit.void))
      const spawner = ChildProcessSpawner.of({
        spawn: () => Effect.sync(() => {
          spawns += 1
          return {
            pid: exact.pid,
            handoff: ownerLock.tryAcquire.pipe(
              Effect.provideService(Scope.Scope, candidateScope),
              Effect.mapError((error) => new AcnEnsuranceFailed({ reason: String(error) })),
              Effect.flatMap((acquired) => Option.match(acquired, {
                onNone: () => Effect.dieMessage("candidate could not acquire owner lock"),
                onSome: (owner) => {
                  return owner.publish({ ...exact, port: server.port! }).pipe(
                    Effect.mapError((error) => new AcnEnsuranceFailed({ reason: String(error) })),
                  )
                },
              })),
            ),
          }
        }),
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
})
