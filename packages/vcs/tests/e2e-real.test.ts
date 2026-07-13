/**
 * E2E VCS verification — exercises the ACTUAL production code path.
 *
 * No mocks, no fake backends. Real filesystem → makeShadowVcsLayer → real just-git.
 */
import { describe, it, expect } from "vitest"
import { Effect, Layer } from "effect"
import * as path from "node:path"
import * as fs from "node:fs/promises"
import { ShadowVcs, makeShadowVcsLayer, VcsFs, selectorToRestoreScope } from "../src/index"
import { realFs } from "../src/backends/just-git"

describe("E2E: real filesystem + real just-git backend", () => {
  async function withVcs<T>(fn: (tmpDir: string, run: <A>(eff: Effect.Effect<A, unknown, ShadowVcs>) => Promise<A>) => Promise<T>): Promise<T> {
    const tmpDir = await fs.mkdtemp("/tmp/vcs-e2e-")
    const storagePath = path.join(tmpDir, ".vcs")
    const vcsLayer = makeShadowVcsLayer({ worktreePath: tmpDir, storagePath }).pipe(
      Layer.provide(Layer.succeed(VcsFs, realFs))
    )
    // Resolve the Vcs service once so undo/redo share the same instance
    const vcsService = await Effect.runPromise(
      Effect.gen(function* () {
        return yield* ShadowVcs
      }).pipe(Effect.provide(vcsLayer))
    )
    const run = <A>(eff: Effect.Effect<A, unknown, ShadowVcs>) =>
      Effect.runPromise(eff.pipe(Effect.provideService(ShadowVcs, vcsService)))
    try {
      return await fn(tmpDir, run)
    } finally {
      await fs.rm(tmpDir, { recursive: true, force: true })
    }
  }

  it("record creates checkpoints, undo/redo restore worktree, scoped restore works", async () => {
    await withVcs(async (tmpDir, run) => {
      // 1. Initial state
      await fs.writeFile(path.join(tmpDir, "foo.txt"), "Hello World")

      // 2. record() creates initial checkpoint
      const cp1 = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.record({ message: "initial" })
      }))
      expect(cp1).toBeTruthy()

      // 3. Modify files
      await fs.writeFile(path.join(tmpDir, "foo.txt"), "Modified")
      await fs.writeFile(path.join(tmpDir, "bar.txt"), "New file")

      // 4. record() creates second checkpoint
      const cp2 = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.record({ message: "second" })
      }))
      expect(cp2).toBeTruthy()
      expect(cp2).not.toBe(cp1)

      // 5. diffWorking detects all changes vs time epoch
      const diff = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.diffWorking({ against: { kind: "time", when: new Date(0) } })
      }))
      expect(diff.files.map(f => f.path).sort()).toEqual(["bar.txt", "foo.txt"])
      expect(diff.additions).toBeGreaterThanOrEqual(1)

      // 6. Three checkpoints exist (initial + two explicit records)
      const checkpoints = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.listCheckpoints()
      }))
      expect(checkpoints.length).toBe(3)
      expect(checkpoints.map(c => c.name)).toEqual(["1", "2", "3"])

      // 7. undo() restores to checkpoint 2 state (first explicit record)
      const undoResult = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.undo()
      }))
      expect(undoResult.restoredOperationId).toBe(cp1)

      const fooAfterUndo = await fs.readFile(path.join(tmpDir, "foo.txt"), "utf-8")
      expect(fooAfterUndo).toBe("Hello World")
      await expect(fs.access(path.join(tmpDir, "bar.txt"))).rejects.toThrow()

      // 8. redo() restores to checkpoint 3 state (second explicit record)
      const redoResult = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.redo()
      }))
      expect(redoResult.restoredOperationId).toBe(cp2)

      const fooAfterRedo = await fs.readFile(path.join(tmpDir, "foo.txt"), "utf-8")
      expect(fooAfterRedo).toBe("Modified")
      const barAfterRedo = await fs.readFile(path.join(tmpDir, "bar.txt"), "utf-8")
      expect(barAfterRedo).toBe("New file")

      // 9. restore() to checkpoint 2 (first explicit record) with full scope
      await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        yield* vcs.restore({ to: { kind: "checkpoint", name: "2" } })
      }))
      const fooAfterFullRestore = await fs.readFile(path.join(tmpDir, "foo.txt"), "utf-8")
      expect(fooAfterFullRestore).toBe("Hello World")
      await expect(fs.access(path.join(tmpDir, "bar.txt"))).rejects.toThrow()

      // 10. restore() scoped to single file (restore to cp3 first, then scoped to cp2)
      await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        yield* vcs.restore({ to: { kind: "checkpoint", name: "3" } })
      }))
      await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        yield* vcs.restore({
          to: { kind: "checkpoint", name: "2" },
          scope: { kind: "file", path: "foo.txt" }
        })
      }))
      const fooScoped = await fs.readFile(path.join(tmpDir, "foo.txt"), "utf-8")
      expect(fooScoped).toBe("Hello World")
      // bar.txt should still exist because restore was scoped to foo.txt
      const barScoped = await fs.readFile(path.join(tmpDir, "bar.txt"), "utf-8")
      expect(barScoped).toBe("New file")

      // 11. restore() with glob scope (restore to cp3 first, then glob to cp2)
      await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        yield* vcs.restore({ to: { kind: "checkpoint", name: "3" } })
      }))
      await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        yield* vcs.restore({
          to: { kind: "checkpoint", name: "2" },
          scope: { kind: "glob", pattern: "*.txt" }
        })
      }))
      const fooGlob = await fs.readFile(path.join(tmpDir, "foo.txt"), "utf-8")
      expect(fooGlob).toBe("Hello World")
      await expect(fs.access(path.join(tmpDir, "bar.txt"))).rejects.toThrow()

      // 12. resolve() with time-based point
      const timeResolved = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.resolve({ kind: "time", when: new Date(Date.now() + 10000) })
      }))
      expect(timeResolved).toBeTruthy() // resolves to some checkpoint since it's in the future

      // 13. readAt()
      const readAtResult = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.readAt({
          point: { kind: "checkpoint", name: "2" },
          paths: ["foo.txt"]
        })
      }))
      const readAtContent = new TextDecoder().decode(readAtResult.get("foo.txt")!)
      expect(readAtContent).toBe("Hello World")

      // 14. head returns latest
      const head = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.head
      }))
      expect(head.operationId).toBeTruthy() // head is a valid checkpoint
    })
  })

  it("diffWorking treats trailing-slash pathFilter as directory scope", async () => {
    await withVcs(async (tmpDir, run) => {
      await fs.writeFile(path.join(tmpDir, "baseline.txt"), "baseline")

      const baseline = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.record({ message: "baseline" })
      }))

      await fs.mkdir(path.join(tmpDir, "demo-dir"), { recursive: true })
      await fs.writeFile(path.join(tmpDir, "demo-dir", "file-a.txt"), "a")
      await fs.writeFile(path.join(tmpDir, "demo-root.txt"), "root")

      const dirtyDiff = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.diffWorking({ against: baseline, pathFilter: "demo-dir/" })
      }))
      expect(dirtyDiff.files.map((file) => file.path)).toEqual(["demo-dir/file-a.txt"])

      await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.record({ message: "after demo changes" })
      }))

      const cleanDiff = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.diffWorking({ against: baseline, pathFilter: "demo-dir/" })
      }))
      expect(cleanDiff.files.map((file) => file.path)).toEqual(["demo-dir/file-a.txt"])
    })
  })

  it("rollback path handles unicode, spaces, deep paths, symlinks, binary, and large files", async () => {
    await withVcs(async (tmpDir, run) => {
      await fs.writeFile(path.join(tmpDir, "baseline.txt"), "baseline")

      const baseline = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.record({ message: "baseline" })
      }))

      const binary = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x00, 0xff, 0xfe])
      const large = new Uint8Array(1024 * 1024)
      large.fill(65)
      large[512] = 0

      await fs.mkdir(path.join(tmpDir, "deep", "nested", "dir"), { recursive: true })
      await fs.mkdir(path.join(tmpDir, "images"), { recursive: true })
      await fs.writeFile(path.join(tmpDir, "simple.txt"), "simple")
      await fs.writeFile(path.join(tmpDir, "ファイル.txt"), "unicode")
      await fs.writeFile(path.join(tmpDir, "space name.txt"), "spaces")
      await fs.writeFile(path.join(tmpDir, "deep", "nested", "dir", "file.txt"), "deep")
      await fs.writeFile(path.join(tmpDir, "images", "interface.png"), binary)
      await fs.writeFile(path.join(tmpDir, "large.bin"), large)
      await fs.symlink("baseline.txt", path.join(tmpDir, "link-to-baseline"))

      const diffBefore = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.diffWorking({ against: baseline, pathFilter: "**/*" })
      }))

      const expectedPaths = [
        "deep/nested/dir/file.txt",
        "images/interface.png",
        "large.bin",
        "link-to-baseline",
        "simple.txt",
        "space name.txt",
        "ファイル.txt",
      ]
      expect(diffBefore.files.map((file) => file.path).sort()).toEqual(expectedPaths)
      expect(diffBefore.files.find((file) => file.path === "images/interface.png")?.diff)
        .toContain("Binary files")

      await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        yield* vcs.restore({ to: baseline, scope: selectorToRestoreScope("**/*") })
      }))

      await expect(fs.readFile(path.join(tmpDir, "baseline.txt"), "utf-8"))
        .resolves.toBe("baseline")
      for (const relPath of expectedPaths) {
        await expect(fs.lstat(path.join(tmpDir, relPath))).rejects.toThrow()
      }

      const diffAfter = await run(Effect.gen(function* () {
        const vcs = yield* ShadowVcs
        return yield* vcs.diffWorking({ against: baseline, pathFilter: "**/*" })
      }))
      expect(diffAfter.files).toEqual([])
    })
  })
})
