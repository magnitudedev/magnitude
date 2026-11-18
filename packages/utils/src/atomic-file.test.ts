import { describe, expect, test } from "vitest"
import * as FileSystem from "@effect/platform/FileSystem"
import { NodeContext } from "@effect/platform-node"
import { Effect, Layer } from "effect"
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { basename, join } from "node:path"
import { writeFileAtomic } from "./atomic-file"

// A file system that cannot rename over an existing file, as Windows cannot, recording each rename.
const windowsLike = (renames: Array<[string, string]>) =>
  Layer.effect(
    FileSystem.FileSystem,
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      return {
        ...fs,
        rename: (from: string, to: string) => Effect.gen(function* () {
          renames.push([from, to])
          if (yield* fs.exists(to)) return yield* fs.rename(from, `${to}/refused`)
          return yield* fs.rename(from, to)
        }),
      }
    }),
  ).pipe(Layer.provide(NodeContext.layer))

describe("writeFileAtomic", () => {
  test("keeps the original aside under a name no temporary-file sweep removes", async () => {
    const file = join(mkdtempSync(join(tmpdir(), "atomic-file-")), "local-models.json")
    writeFileSync(file, "old")
    const renames: Array<[string, string]> = []
    await Effect.runPromise(writeFileAtomic(file, "new").pipe(
      Effect.provide(windowsLike(renames)),
      Effect.provide(NodeContext.layer),
    ) as Effect.Effect<void>)
    expect(readFileSync(file, "utf8")).toBe("new")
    const aside = renames.find(([from]) => from === file)?.[1]
    expect(aside).toBeDefined()
    // The stores' sweeps remove `.<name>.<pid>.<unique>.tmp` once abandoned.
    expect(basename(aside!)).not.toMatch(/^\..*\.tmp$/)
  })
})
