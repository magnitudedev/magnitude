import * as HttpServer from "@effect/platform/HttpServer"
import { BunCommandExecutor, BunFileSystem } from "@effect/platform-bun"
import { Context, Effect, Exit, Layer, Option, Scope } from "effect"
import { describe, expect, it } from "vitest"
import { AcnOwnerIdSchema } from "@magnitudedev/acn-protocol"
import { mkdtemp } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { AcnServiceLifecycle, AcnServiceLifecycleLive } from "./service-lifecycle"
import {
  readRegistration,
  registrationPath,
  writeRegistrationAtomic,
} from "./daemon-registration"
import { DaemonLifecycleLive } from "./daemon-lifecycle"
import { currentProcessStartIdentity } from "./process-identity"
import { ACN_OWNER_ID } from "./identity"

const fakeServer = HttpServer.make({
  address: {
    _tag: "TcpAddress",
    hostname: "127.0.0.1",
    port: 43210,
  },
  serve: () => Effect.never,
})

describe("ACN daemon lifecycle", () => {
  it("self-retires after canonical ownership changes and preserves the successor", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-acn-lifecycle-"))
    const path = registrationPath(dataDir)

    await Effect.gen(function* () {
      const scope = yield* Scope.make()
      const lifecycleContext = yield* Layer.buildWithScope(
        AcnServiceLifecycleLive(),
        scope
      )
      const lifecycle = Context.get(lifecycleContext, AcnServiceLifecycle)
      const processStartIdentity = yield* currentProcessStartIdentity.pipe(
        Effect.provide(
          BunCommandExecutor.layer.pipe(Layer.provide(BunFileSystem.layer)),
        ),
      )
      const dependencies = Layer.mergeAll(
        Layer.succeed(AcnServiceLifecycle, lifecycle),
        BunFileSystem.layer,
        BunCommandExecutor.layer.pipe(Layer.provide(BunFileSystem.layer)),
        Layer.succeed(HttpServer.HttpServer, fakeServer)
      )
      yield* Layer.buildWithScope(
        DaemonLifecycleLive({
          version: "1.0.0",
          register: true,
          debug: false,
          dataDir,
          instance: {
            path: join(dataDir, "acn", "instances", "test.json"),
            record: {
              id: ACN_OWNER_ID,
              version: "1.0.0",
              url: Option.none(),
              pid: process.pid,
              processStartIdentity,
            },
          },
          ownershipCheckIntervalMs: 10,
          ownershipCheckTimeoutMs: 100,
        }),
        scope
      ).pipe(Effect.provide(dependencies))

      yield* writeRegistrationAtomic(path, {
        id: AcnOwnerIdSchema.make("successor"),
        version: "2.0.0",
        url: "http://127.0.0.1:43211",
        pid: process.pid,
        timestamp: Date.now(),
      })
      const request = yield* lifecycle.awaitStopping.pipe(Effect.timeout("1 second"))
      expect(request.reason).toBe("ownership-lost")

      yield* Scope.close(scope, Exit.void)
      expect(yield* readRegistration(path)).toEqual(
        Option.some(
          expect.objectContaining({
            id: "successor",
            version: "2.0.0",
          })
        )
      )
    }).pipe(Effect.provide(BunFileSystem.layer), Effect.runPromise)
  })
})
