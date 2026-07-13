/**
 * REPRO: record() doesn't create new checkpoints after file changes.
 *
 * The initial checkpoint (from makeShadowVcs) works.
 * But subsequent vcs.record() calls after file changes don't create new checkpoints.
 * This is the root cause of "No checkpoint before given time" on undo.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { Effect, Layer } from "effect"
import { makeShadowVcsLayer, ShadowVcs, VcsFsLive } from "../src/index"
import { mkdir, rm, writeFile, readFile } from "node:fs/promises"
import * as path from "node:path"
import * as os from "node:os"

describe("REPRO: record() doesn't create checkpoints after changes", () => {
  let projectDir: string
  let storagePath: string

  beforeEach(async () => {
    projectDir = os.tmpdir() + "/repro-record-" + Date.now()
    storagePath = path.join(projectDir, ".magnitude", ".vcs")
    await mkdir(projectDir, { recursive: true })
    await mkdir(path.join(projectDir, "src"), { recursive: true })
    await writeFile(path.join(projectDir, "foo.txt"), "Hello World\n")
  })

  afterEach(async () => {
    await rm(projectDir, { recursive: true, force: true }).catch(() => {})
  })

  it("record() creates checkpoint when changes exist (via ShadowVcs layer)", async () => {
    const vcsLayer = makeShadowVcsLayer({
      worktreePath: projectDir,
      storagePath,
    }).pipe(Layer.provide(VcsFsLive))

    const forkLayer = Layer.mergeAll(vcsLayer)

    await Effect.runPromise(
      Effect.gen(function* () {
        const vcs = yield* ShadowVcs

        // Check initial state
        const head1 = yield* vcs.head
        console.log("Head after init:", head1.commitHash.slice(0, 7))

        const cps1 = yield* vcs.listCheckpoints()
        console.log("Checkpoints after init:", cps1.length, cps1.map(c => c.name))

        // Modify file
        yield* Effect.promise(() =>
          writeFile(path.join(projectDir, "foo.txt"), "Modified!\n")
        )

        // Record
        const opId = yield* vcs.record({ message: "modified foo" })
        console.log("record() returned:", opId.slice(0, 7))

        // Check checkpoints
        const cps2 = yield* vcs.listCheckpoints()
        console.log("Checkpoints after record:", cps2.length, cps2.map(c => c.name))

        // Check head
        const head2 = yield* vcs.head
        console.log("Head after record:", head2.commitHash.slice(0, 7))

        // Is the file content in the head commit?
        const readResult = yield* vcs.readAt({
          point: head2.commitHash,
          paths: ["foo.txt"],
        })
        const content = readResult.get("foo.txt")
        console.log("foo.txt at HEAD:", content ? new TextDecoder().decode(content) : "NOT FOUND")

        // Verify actual file
        const actualContent = yield* Effect.promise(() =>
          readFile(path.join(projectDir, "foo.txt"), "utf8")
        )
        console.log("foo.txt on disk:", actualContent)

        expect(cps2.length).toBeGreaterThan(cps1.length)
      }).pipe(Effect.provide(forkLayer))
    )
  })
})
