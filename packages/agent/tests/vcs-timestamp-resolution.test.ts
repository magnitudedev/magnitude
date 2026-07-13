import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { Effect, Layer } from "effect"
import * as path from "path"
import * as fs from "fs/promises"
import { ShadowVcs, makeShadowVcsLayer, VcsFsLive } from "@magnitudedev/vcs"

const tmpDir = path.join(process.cwd(), "tmp-vcs-timestamp-test")

async function setup() {
  await fs.mkdir(tmpDir, { recursive: true })
  await fs.rm(path.join(tmpDir, ".vcs"), { recursive: true, force: true })
  await fs.rm(path.join(tmpDir, "foo.txt"), { recursive: true, force: true })
}

async function teardown() {
  await fs.rm(tmpDir, { recursive: true, force: true })
}

describe("VCS time-based resolution", () => {
  beforeEach(setup)
  afterEach(teardown)

  it("resolves timestamps to the most recent checkpoint at or before that time", async () => {
    const worktreePath = tmpDir
    const vcsLayer = makeShadowVcsLayer({
      worktreePath,
      storagePath: path.join(worktreePath, ".vcs"),
      timezone: "UTC",
    }).pipe(Layer.provide(VcsFsLive))

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const vcs = yield* ShadowVcs

        // Record initial checkpoint (empty state)
        const initialOp = yield* vcs.record()
        const initCheck = yield* vcs.getCheckpoint(initialOp)
        const initialTimestamp = initCheck.timestamp.getTime()

        // Immediately write a file and record another checkpoint.
        // Git timestamps have 1-second resolution, so these two checkpoints
        // will almost certainly share the same git second.
        yield* Effect.promise(() => fs.writeFile(path.join(worktreePath, "foo.txt"), "Hello"))

        const toolOp = yield* vcs.record()
        const toolCheck = yield* vcs.getCheckpoint(toolOp)
        const toolTimestamp = toolCheck.timestamp.getTime()

        // --- Key test: when both checkpoints share the same git second,
        // resolving the exact second should return the OLDEST one (initial),
        // because the user means "restore to the state at the start of that
        // second" — i.e. before any tools ran in this turn.
        //
        // If we naively used `find(c => ts <= target)` (first match = newest),
        // we'd get the tool checkpoint and undo would do nothing.
        const sharedSecondTimestamp =
          initialTimestamp === toolTimestamp ? initialTimestamp : null

        const resolvedShared = sharedSecondTimestamp !== null
          ? yield* vcs.resolve({ kind: "time", when: new Date(sharedSecondTimestamp) })
          : null

        // --- For distinct seconds, resolving the tool second should return tool ---
        const resolvedTool = yield* vcs.resolve({
          kind: "time",
          when: new Date(toolTimestamp),
        })

        // --- Undo using the shared-second timestamp should remove foo.txt ---
        const undoResult = sharedSecondTimestamp !== null
          ? yield* Effect.either(
              vcs.restore({ to: { kind: "time", when: new Date(sharedSecondTimestamp) } })
            )
          : null

        const existsAfterUndo = yield* Effect.tryPromise({
          try: async () => {
            try {
              await fs.access(path.join(worktreePath, "foo.txt"))
              return true
            } catch {
              return false
            }
          },
          catch: () => false,
        })

        return {
          sharedSecondTimestamp,
          initialOp,
          resolvedShared,
          toolOp,
          resolvedTool,
          undoResult,
          existsAfterUndo,
        }
      }).pipe(Effect.provide(vcsLayer)),
    )

    // If the two checkpoints happened in the same git second (the real bug scenario),
    // resolving that shared second must return the initial checkpoint (oldest).
    // That is the state the user means when they say "since 17:10:01" — the
    // state before any tools ran in that turn.
    if (result.sharedSecondTimestamp !== null) {
      expect(result.resolvedShared).toBe(result.initialOp)

      // Resolving the tool checkpoint's exact timestamp ALSO returns the initial
      // checkpoint, because both share the same git second and we pick the oldest.
      expect(result.resolvedTool).toBe(result.initialOp)

      // Restore should have succeeded and removed foo.txt
      expect(result.undoResult).toBeTruthy()
      expect(result.undoResult!._tag).not.toBe("Left")
      expect(result.existsAfterUndo).toBe(false)
    }
  })
})
