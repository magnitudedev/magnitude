/**
 * REPRO: VCS record() never runs in the harness afterExecute hook.
 *
 * Bug: After tool execution, afterExecute calls yield* ShadowVcs then vcs.record().
 * If ShadowVcs isn't in the fork layer's Effect context, it throws "Service not found",
 * which catchAllCause swallows silently. No checkpoint is ever recorded.
 * User sees "No checkpoint before given time" on undo/rollback.
 *
 * This test reproduces the exact DI flow:
 * 1. makeShadowVcsLayer → vcsLayer
 * 2. makeForkLayers → forkLayer (merges vcsLayer)
 * 3. buildStandardHooks → afterExecute (yields* ShadowVcs)
 * 4. provideLayer(effect, forkLayer) — does ShadowVcs actually reach the hook?
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { Effect, Layer, Cause } from "effect"
import { makeShadowVcsLayer, ShadowVcs, VcsFsLive } from "../src/index"
import { mkdir, rm, writeFile } from "node:fs/promises"
import * as path from "node:path"
import * as os from "node:os"

// ── Simulated harness provideLayer (from dispatcher.ts) ─────────────
// This cast hides missing services — the core of the bug.
function provideLayer<A, E>(effect: Effect.Effect<A, E, unknown>): Effect.Effect<A, E, never> {
  return effect as Effect.Effect<A, E, never>
}

describe("REPRO: harness afterExecute VCS integration", () => {
  let projectDir: string
  let storagePath: string

  beforeEach(async () => {
    projectDir = os.tmpdir() + "/repro-harness-vcs-" + Date.now()
    storagePath = path.join(projectDir, ".magnitude", ".vcs")
    await mkdir(projectDir, { recursive: true })
    await mkdir(path.join(projectDir, "src"), { recursive: true })
    await writeFile(path.join(projectDir, "foo.txt"), "Hello World\n")
  })

  afterEach(async () => {
    await rm(projectDir, { recursive: true, force: true }).catch(() => {})
  })

  it("afterExecute CAN access ShadowVcs when fork layer includes vcsLayer", async () => {
    const vcsLayer = makeShadowVcsLayer({
      worktreePath: projectDir,
      storagePath,
    }).pipe(Layer.provide(VcsFsLive))

    // Simulates makeForkLayers — merges vcsLayer into the fork layer
    const forkLayer = Layer.mergeAll(vcsLayer)

    // Simulates afterExecute hook
    const afterExecute = Effect.gen(function* () {
      const vcs = yield* ShadowVcs
      const opId = yield* vcs.record({ message: "tool-call-1" })
      return opId
    })

    // Simulates dispatcher's provideLayer + Effect.provide
    const result = await Effect.runPromiseExit(
      provideLayer(afterExecute).pipe(Effect.provide(forkLayer))
    )

    expect(result._tag).toBe("Success")
    if (result._tag === "Success") {
      expect(result.value).toBeTruthy()
    }
  })

  it("afterExecute FAILS SILENTLY when fork layer does NOT include vcsLayer", async () => {
    // No VCS layer — simulates the bug where vcsLayer wasn't merged
    const emptyLayer = Layer.empty

    const afterExecute = Effect.gen(function* () {
      const vcs = yield* ShadowVcs
      yield* vcs.record({ message: "tool-call-1" })
    })

    // Without catchAllCause: properly fails with Service not found
    const result = await Effect.runPromiseExit(
      provideLayer(afterExecute).pipe(Effect.provide(emptyLayer))
    )
    expect(result._tag).toBe("Failure")

    // With catchAllCause (harness behavior): error is SILENTLY SWALLOWED
    // This is exactly what the user experiences — no error, no checkpoint
    const swallowed = await Effect.runPromiseExit(
      provideLayer(afterExecute).pipe(
        Effect.provide(emptyLayer),
        Effect.catchAllCause(() => Effect.succeed("swallowed")),
      )
    )
    expect(swallowed._tag).toBe("Success")
    if (swallowed._tag === "Success") {
      expect(swallowed.value).toBe("swallowed")
    }
  })

  it("full flow: record + undo actually restores files", async () => {
    const vcsLayer = makeShadowVcsLayer({
      worktreePath: projectDir,
      storagePath,
    }).pipe(Layer.provide(VcsFsLive))

    const forkLayer = Layer.mergeAll(vcsLayer)

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const vcs = yield* ShadowVcs

        // Record initial
        yield* vcs.record({ message: "initial" })

        // Modify file
        yield* Effect.promise(() =>
          writeFile(path.join(projectDir, "foo.txt"), "Modified!\n")
        )

        // Record change
        yield* vcs.record({ message: "modified" })

        // Undo
        const undoResult = yield* vcs.undo()

        // Read file — should be back to "Hello World\n"
        const content = yield* Effect.promise(() =>
          import("node:fs/promises").then((fs) => fs.readFile(path.join(projectDir, "foo.txt"), "utf8"))
        )

        return { undoResult, content }
      }).pipe(Effect.provide(forkLayer))
    )

    expect(result.content).toBe("Hello World\n")
  })

  it("multiple records create multiple checkpoints", async () => {
    const vcsLayer = makeShadowVcsLayer({
      worktreePath: projectDir,
      storagePath,
    }).pipe(Layer.provide(VcsFsLive))

    const forkLayer = Layer.mergeAll(vcsLayer)

    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const vcs = yield* ShadowVcs

        // Initial is checkpoint 1 (from makeShadowVcs)
        // record() with no changes returns HEAD, no new checkpoint
        yield* vcs.record({ message: "no-op" })

        // Modify and record — creates checkpoint 2
        yield* Effect.promise(() =>
          writeFile(path.join(projectDir, "bar.txt"), "bar\n")
        )
        yield* vcs.record({ message: "added bar" })

        // Modify again — creates checkpoint 3
        yield* Effect.promise(() =>
          writeFile(path.join(projectDir, "baz.txt"), "baz\n")
        )
        yield* vcs.record({ message: "added baz" })

        const cps = yield* vcs.listCheckpoints()
        return cps
      }).pipe(Effect.provide(forkLayer))
    )

    // Checkpoint 1 (init) + 2 (bar) + 3 (baz) = 3
    expect(result.length).toBe(3)
  })
})
