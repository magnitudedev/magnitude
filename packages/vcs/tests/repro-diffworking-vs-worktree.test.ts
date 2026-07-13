import { describe, test, expect } from "vitest"
import { Effect } from "effect"
import * as fs from "node:fs/promises"
import * as path from "node:path"
import { createJustGitBackend, realFs } from "../src/backends/just-git"
import { buildShadowVcs } from "../src/layer"

describe("REPRO: diffWorking must detect uncommitted worktree changes", () => {
  test("diffWorking(against=HEAD) shows worktree changes not yet committed", async () => {
    const tmpDir = await fs.mkdtemp("/tmp/repro-diffworking-worktree-")

    // Create initial worktree file
    await fs.writeFile(path.join(tmpDir, "test.txt"), "ORIGINAL")

    const gitDirPath = path.join(tmpDir, ".shadow", ".git")
    await fs.mkdir(gitDirPath, { recursive: true })

    const backend = await createJustGitBackend(tmpDir, gitDirPath, realFs)
    const vcs = await buildShadowVcs(backend, tmpDir)

    // Record initial checkpoint (captures test.txt = "ORIGINAL")
    const id1 = await Effect.runPromise(vcs.record())
    console.log("Checkpoint 1:", id1)

    // NOW: simulate agent making file changes WITHOUT calling record()
    await fs.writeFile(path.join(tmpDir, "test.txt"), "MODIFIED")

    // diffWorking against the checkpoint SHOULD show the uncommitted change
    const diff = await Effect.runPromise(vcs.diffWorking({ against: id1 }))
    console.log("Diff against checkpoint 1:", JSON.stringify(diff, null, 2))

    // This is the BUG: diffWorking returns 0 because it compares
    // againstCommit.tree vs HEAD.tree, both of which are still "ORIGINAL"
    // The worktree change "MODIFIED" is NOT reflected in HEAD
    expect(diff.files.length).toBe(1)
    expect(diff.files[0]!.path).toBe("test.txt")
    expect(diff.files[0]!.status).toBe("modified")
  })
})
