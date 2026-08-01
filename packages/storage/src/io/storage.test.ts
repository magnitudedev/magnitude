import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import * as FileSystem from "@effect/platform/FileSystem"
import { SystemError } from "@effect/platform/Error"
import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Deferred, Effect, Fiber, Layer } from "effect"
import { afterEach, beforeEach, describe, expect, test } from "vitest"

import { makeStorageIo } from "./storage"

describe("recoverable JSONL storage", () => {
  let tmpDir: string

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "magnitude-jsonl-storage-"))
  })

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true })
  })

  test("append mutation completes before fiber interruption is observed", async () => {
    const filePath = join(tmpDir, "events.jsonl")

    const exit = await Effect.runPromise(
      Effect.gen(function* () {
        const baseFs = yield* FileSystem.FileSystem
        const writeStarted = yield* Deferred.make<void>()
        const delayedFs: FileSystem.FileSystem = {
          ...baseFs,
          writeFileString: (path, data, options) =>
            Deferred.succeed(writeStarted, undefined).pipe(
              Effect.zipRight(Effect.sleep("50 millis")),
              Effect.zipRight(baseFs.writeFileString(path, data, options))
            ),
        }
        const io = yield* makeStorageIo().pipe(
          Effect.provideService(FileSystem.FileSystem, delayedFs)
        )
        const fiber = yield* io
          .recoverableJsonLines<{ readonly type: string }>(filePath)
          .append([{ type: "committed" }])
          .pipe(Effect.fork)

        yield* Deferred.await(writeStarted)
        return yield* Fiber.interrupt(fiber)
      }).pipe(Effect.provide(Layer.merge(BunFileSystem.layer, BunPath.layer)))
    )

    expect(exit._tag).toBe("Failure")
    expect(await readFile(filePath, "utf-8")).toBe('{"type":"committed"}\n')
  })

  test("tail truncation completes before read interruption is observed", async () => {
    const filePath = join(tmpDir, "events.jsonl")
    await writeFile(filePath, '{"type":"committed"}\n{"type":', "utf-8")

    const exit = await Effect.runPromise(
      Effect.gen(function* () {
        const baseFs = yield* FileSystem.FileSystem
        const truncateStarted = yield* Deferred.make<void>()
        const delayedFs: FileSystem.FileSystem = {
          ...baseFs,
          truncate: (path, length) =>
            Deferred.succeed(truncateStarted, undefined).pipe(
              Effect.zipRight(Effect.sleep("50 millis")),
              Effect.zipRight(baseFs.truncate(path, length))
            ),
        }
        const io = yield* makeStorageIo().pipe(
          Effect.provideService(FileSystem.FileSystem, delayedFs)
        )
        const fiber = yield* io
          .recoverableJsonLines<{ readonly type: string }>(filePath)
          .readAndRepair.pipe(Effect.fork)

        yield* Deferred.await(truncateStarted)
        return yield* Fiber.interrupt(fiber)
      }).pipe(Effect.provide(Layer.merge(BunFileSystem.layer, BunPath.layer)))
    )

    expect(exit._tag).toBe("Failure")
    expect(await readFile(filePath, "utf-8")).toBe('{"type":"committed"}\n')
  })

  test("rolls back a partially written append after a filesystem error", async () => {
    const filePath = join(tmpDir, "events.jsonl")

    const exit = await Effect.runPromise(
      Effect.gen(function* () {
        const baseFs = yield* FileSystem.FileSystem
        const failingFs: FileSystem.FileSystem = {
          ...baseFs,
          writeFileString: (path, _data, options) =>
            baseFs
              .writeFileString(path, '{"type":', options)
              .pipe(
                Effect.zipRight(
                  new SystemError({
                    reason: "WriteZero",
                    module: "FileSystem",
                    method: "writeFileString",
                    pathOrDescriptor: path,
                  })
                )
              ),
        }
        const io = yield* makeStorageIo().pipe(
          Effect.provideService(FileSystem.FileSystem, failingFs)
        )
        return yield* Effect.exit(
          io
            .recoverableJsonLines<{ readonly type: string }>(filePath)
            .append([{ type: "not-committed" }])
        )
      }).pipe(Effect.provide(Layer.merge(BunFileSystem.layer, BunPath.layer)))
    )

    expect(exit._tag).toBe("Failure")
    expect(await readFile(filePath, "utf-8")).toBe("")
  })
})
