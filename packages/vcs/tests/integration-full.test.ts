/**
 * Integration test: exercises the REAL ShadowVcs layer + just-git backend
 * against a real filesystem — same code path the product uses.
 */
import { Effect, Layer } from "effect"
import { ShadowVcs, makeShadowVcsLayer, VcsFs } from "../src/index"
import { realFs } from "../src/backends/just-git"
import * as path from "node:path"
import * as fs from "node:fs/promises"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { describe, it, expect, beforeAll, afterAll } from "vitest"

let tmpDir: string
let worktreePath: string
let storagePath: string

beforeAll(() => {
  tmpDir = mkdtempSync(path.join(tmpdir(), "vcs-int-"))
  worktreePath = path.join(tmpDir, "project")
  storagePath = path.join(tmpDir, ".vcs")
})

afterAll(() => {
  rmSync(tmpDir, { recursive: true, force: true })
})

describe("ShadowVcs integration (real filesystem)", () => {
  const vcsLayer = () =>
    makeShadowVcsLayer({ worktreePath, storagePath }).pipe(
      Layer.provide(Layer.succeed(VcsFs, realFs)),
    )

  const run = <A>(eff: Effect.Effect<A, any, ShadowVcs>) =>
    Effect.runPromise(eff.pipe(Effect.provide(vcsLayer())))

  const getVcs = () => run(Effect.gen(function* () { return yield* ShadowVcs }))

  it("full rollback cycle: record, modify, diffWorking, restore, verify", async () => {
    await fs.mkdir(worktreePath, { recursive: true })

    // Step 1: Create initial files + checkpoint
    await fs.writeFile(path.join(worktreePath, "hello.py"), "def greet(name):\n    return f'Hello {name}'\n")
    await fs.writeFile(path.join(worktreePath, "config.json"), '{"version": "1.0.0", "debug": false}')
    await fs.mkdir(path.join(worktreePath, "src"), { recursive: true })
    await fs.writeFile(path.join(worktreePath, "src", "main.py"), "print('hello')")

    const vcs = await getVcs()
    const cp1 = await run(vcs.record({ message: "initial" }))
    expect(cp1).toBeTruthy()

    // Step 2: Make changes
    await fs.writeFile(path.join(worktreePath, "hello.py"), "def greet(name, excited=False):\n    suffix = '!' if excited else ''\n    return f'Hello {name}{suffix}'\n\ndef farewell(name):\n    return f'Goodbye {name}'\n")
    await fs.writeFile(path.join(worktreePath, "config.json"), '{"version": "2.0.0", "debug": true, "log_level": "verbose"}')
    await fs.writeFile(path.join(worktreePath, "utils.py"), "# utility functions\n")

    const cp2 = await run(vcs.record({ message: "modifications" }))
    expect(cp2).toBeTruthy()

    // Step 3: diff cp1 vs cp2 using the diff() API (diffWorking compares vs HEAD)
    const diff1 = await run(vcs.diff({ from: cp1, to: cp2, pathFilter: "**/*" }))
    expect(diff1.files.length).toBeGreaterThan(0)
    expect(diff1.modifications).toBeGreaterThanOrEqual(2)
    expect(diff1.additions).toBeGreaterThanOrEqual(1)

    // Step 4: diff with pathFilter
    const diff2 = await run(vcs.diff({ from: cp1, to: cp2, pathFilter: "*.py" }))
    expect(diff2.files.length).toBeGreaterThan(0)

    // Step 5: diff without filter
    const diff3 = await run(vcs.diff({ from: cp1, to: cp2 }))
    expect(diff3.files.length).toBeGreaterThanOrEqual(3)

    // Step 6: Get diff BEFORE restore (same as checkpoint_rollback tool)
    const diffBefore = await run(vcs.diff({ from: cp1, to: cp2, pathFilter: "**/*" }))
    const filesRolledBack = diffBefore.files.map(f => f.path).join(", ") || "none"
    expect(filesRolledBack).not.toBe("none")

    // Step 7: Restore
    const result = await run(vcs.restore({ to: cp1, scope: { kind: "full" } }))
    expect(result.targetSnapshotId).toBeTruthy()

    // Step 8: Verify files are restored
    const helloContent = await fs.readFile(path.join(worktreePath, "hello.py"), "utf-8")
    expect(helloContent).toContain("def greet(name):")
    expect(helloContent).not.toContain("farewell")

    const configContent = await fs.readFile(path.join(worktreePath, "config.json"), "utf-8")
    expect(configContent).toContain('"1.0.0"')
    expect(configContent).not.toContain("log_level")

    const utilsExists = await fs.access(path.join(worktreePath, "utils.py")).then(() => true).catch(() => false)
    expect(utilsExists).toBe(false)

    // Step 9: Post-restore checkpoint should have real content
    const cp3 = await run(vcs.record({ message: "after-restore" }))
    expect(cp3).toBeTruthy()

    // After restore + record, worktree matches HEAD, so diffWorking is empty
    const diffAfterRestore = await run(vcs.diffWorking({ against: cp3 }))
    expect(diffAfterRestore.files.length).toBe(0)

    // Step 10: Diff between cp1 and cp2
    const diffCp = await run(vcs.diff({ from: cp1, to: cp2 }))
    expect(diffCp.modifications).toBeGreaterThanOrEqual(2)
  })
})
