/**
 * E2E REPRO: VCS checkpoint flow - proves undo/rollback works.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { Effect, Layer, Cause } from "effect"
import {
  makeShadowVcsLayer,
  ShadowVcs,
  VcsFsLive,
} from "@magnitudedev/vcs"
import { mkdir, rm, writeFile, readFile, stat } from "node:fs/promises"
import * as path from "node:path"
import * as os from "node:os"

describe("E2E: VCS checkpoint flow", () => {
  let projectDir: string
  let scratchpadPath: string

  beforeEach(async () => {
    projectDir = os.tmpdir() + "/e2e-vcs-" + Date.now()
    scratchpadPath = path.join(projectDir, ".magnitude")
    await mkdir(projectDir, { recursive: true })
    await writeFile(path.join(projectDir, "README.md"), "# test\n")
  })

  afterEach(async () => {
    await rm(projectDir, { recursive: true, force: true }).catch(() => {})
  })

  it("record creates checkpoints, undo restores files", async () => {
    const vcsLayer = makeShadowVcsLayer({
      worktreePath: projectDir,
      storagePath: path.join(scratchpadPath, ".vcs"),
      timezone: "UTC",
    }).pipe(Layer.provide(VcsFsLive))

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const vcs = yield* ShadowVcs

        // Check initial state
        const isClean1 = yield* vcs.isClean
        console.log("Initial isClean:", isClean1)

        const changedSince1 = yield* vcs.changedSinceHead
        console.log("Initial changedSinceHead:", changedSince1)

        const head1 = yield* vcs.head
        console.log("Initial head:", head1.commitHash.slice(0,7))

        // Write file
        yield* Effect.promise(() =>
          writeFile(path.join(projectDir, "foo.txt"), "Hello World\n")
        )

        // Verify file exists on disk
        const exists = yield* Effect.promise(() =>
          stat(path.join(projectDir, "foo.txt")).then(() => true, () => false)
        )
        console.log("foo.txt exists:", exists)

        // Check if VCS sees changes
        const isClean2 = yield* vcs.isClean
        console.log("After write isClean:", isClean2)

        const changedSince2 = yield* vcs.changedSinceHead
        console.log("After write changedSinceHead:", changedSince2)

        // Record
        const op1 = yield* vcs.record({ message: "tool-call-1" })
        console.log("record 1 returned:", op1.slice(0,7))

        // Check checkpoints
        const cps2 = yield* vcs.listCheckpoints()
        console.log("After record 1 - checkpoints:", cps2.length, cps2.map(c => ({ name: c.name, hash: c.commitHash.slice(0,7) })))

        // Check if HEAD changed
        const head2 = yield* vcs.head
        console.log("After record 1 - head:", head2.commitHash.slice(0,7))

        // Did record actually create a NEW commit or just return the existing one?
        const headChanged = head1.commitHash !== head2.commitHash
        console.log("HEAD changed after record:", headChanged)

        // Modify file
        yield* Effect.promise(() =>
          writeFile(path.join(projectDir, "foo.txt"), "Modified!\n")
        )

        // Record again
        const op2 = yield* vcs.record({ message: "tool-call-2" })
        console.log("record 2 returned:", op2.slice(0,7))

        const head3 = yield* vcs.head
        console.log("After record 2 - head:", head3.commitHash.slice(0,7))

        const cps3 = yield* vcs.listCheckpoints()
        console.log("After record 2 - checkpoints:", cps3.length, cps3.map(c => ({ name: c.name, hash: c.commitHash.slice(0,7) })))

        // Undo
        const undoResult = yield* Effect.exit(vcs.undo())
        console.log("undo:", undoResult._tag)
        if (undoResult._tag === "Failure") {
          console.log("UNDO FAILED:", Cause.pretty(undoResult.cause).slice(0, 500))
        }
        expect(undoResult._tag).toBe("Success")

        // Verify file restored
        const afterUndo = yield* Effect.promise(() =>
          readFile(path.join(projectDir, "foo.txt"), "utf8")
        )
        console.log("After undo:", afterUndo)
        expect(afterUndo).toBe("Hello World\n")

        return true
      }).pipe(Effect.provide(vcsLayer))
    )

    expect(result).toBe(true)
  })
})
