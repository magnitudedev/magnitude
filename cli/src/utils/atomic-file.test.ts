import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { writeFileAtomic } from "@magnitudedev/utils/atomic-file"

describe("writeFileAtomic", () => {
  it("creates and replaces a configuration without leaving work files", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-atomic-file-" })
      const file = `${directory}/nested/config.json`
      yield* writeFileAtomic(file, "first")
      yield* writeFileAtomic(file, "second")
      return {
        contents: yield* fs.readFileString(file),
        entries: yield* fs.readDirectory(`${directory}/nested`),
      }
    }).pipe(Effect.provide(BunContext.layer))))

    expect(result).toEqual({ contents: "second", entries: ["config.json"] })
  })
})
