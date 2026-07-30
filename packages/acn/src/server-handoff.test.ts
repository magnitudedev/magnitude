import { BunFileSystem } from "@effect/platform-bun"
import { AcnOwnerIdSchema } from "@magnitudedev/acn-protocol"
import { Effect, Fiber, Option, Schedule } from "effect"
import { describe, expect, it } from "vitest"
import { mkdir, mkdtemp, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import {
  readRegistration,
  registrationPath,
  writeRegistrationAtomic,
} from "./daemon-registration"
import { launchAcnServer } from "./server"

describe("ACN server ownership handoff", () => {
  it("publishes its startup server before waiting for active ownership", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-acn-handoff-"))
    const ownerPath = join(dataDir, "acn", "owner")
    await mkdir(join(dataDir, "acn"), { recursive: true })
    await writeFile(
      ownerPath,
      JSON.stringify({
        id: "predecessor",
        pid: process.pid,
        version: "old",
        startedAt: Date.now(),
      })
    )

    await Effect.gen(function* () {
      const server = yield* launchAcnServer({
        register: true,
        dataDir,
      }).pipe(Effect.fork)
      const registration = yield* readRegistration(
        registrationPath(dataDir)
      ).pipe(
        Effect.filterOrFail(Option.isSome, () => "registration not published"),
        Effect.map((value) => value.value),
        Effect.retry({
          schedule: Schedule.spaced("25 millis"),
          while: (error) => error === "registration not published",
        }),
        Effect.timeout("2 seconds")
      )

      const health = yield* Effect.tryPromise(() =>
        fetch(`${registration.url}/health`).then((response) => response.json())
      )
      expect(health).toEqual(
        expect.objectContaining({
          id: registration.id,
          state: {
            _tag: "Starting",
            activity: "WaitingForOwnership",
          },
        })
      )
      const ownerBeforeRetirement = yield* Effect.promise(() =>
        Bun.file(ownerPath).json()
      )
      expect(ownerBeforeRetirement.id).toBe("predecessor")

      yield* writeRegistrationAtomic(registrationPath(dataDir), {
        id: AcnOwnerIdSchema.make("successor"),
        version: "new",
        url: "http://127.0.0.1:43211",
        pid: process.pid,
        timestamp: Date.now(),
      })
      yield* Fiber.join(server).pipe(Effect.timeout("2 seconds"))
      const ownerAfterRetirement = yield* Effect.promise(() =>
        Bun.file(ownerPath).json()
      )
      expect(ownerAfterRetirement.id).toBe("predecessor")
    }).pipe(Effect.provide(BunFileSystem.layer), Effect.runPromise)
  })
})
