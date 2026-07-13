import { describe, test, expect, beforeAll } from "vitest"
import { Effect } from "effect"
import * as fs from "node:fs/promises"
import * as path from "node:path"
import { execSync } from "node:child_process"
import { createJustGitBackend, realFs } from "../src/backends/just-git"
import { buildShadowVcs } from "../src/layer"

describe("REPRO: shadow VCS in real git repo", () => {
  let tmpDir: string

  beforeAll(async () => {
    tmpDir = await fs.mkdtemp("/tmp/repro-shadow-in-real-git-")

    // Initialize a REAL user git repo
    execSync("git init", { cwd: tmpDir })
    execSync("git config user.email 'test@test.com'", { cwd: tmpDir })
    execSync("git config user.name 'Test'", { cwd: tmpDir })

    // Create and commit a file in the user repo
    await fs.mkdir(path.join(tmpDir, "src"), { recursive: true })
    await fs.writeFile(path.join(tmpDir, "user-file.txt"), "original user content")
    execSync("git add .", { cwd: tmpDir })
    execSync("git commit -m 'initial user commit'", { cwd: tmpDir })

    // Create agent worktree file
    await fs.writeFile(path.join(tmpDir, "test.txt"), "ORIGINAL")
  })

  test("getChangedFiles detects modifications when user's .git exists", async () => {
    const gitDirPath = path.join(tmpDir, ".shadow", ".git")
    await fs.mkdir(gitDirPath, { recursive: true })

    const backend = await createJustGitBackend(tmpDir, gitDirPath, realFs)
    const vcs = await buildShadowVcs(backend, tmpDir)

    // Record initial checkpoint (should capture test.txt = "ORIGINAL")
    const id1 = await Effect.runPromise(vcs.record())
    console.log("Initial checkpoint:", id1)

    // Modify file (simulating agent edit)
    await fs.writeFile(path.join(tmpDir, "test.txt"), "MODIFIED")

    // This should detect the change
    const changed = await Effect.runPromise(
      Effect.gen(function* () {
        return yield* vcs.changedSinceHead
      })
    )
    console.log("Changed since head:", changed)
    expect(changed).toContain("test.txt")

    // Record the change
    const id2 = await Effect.runPromise(vcs.record())
    console.log("Second checkpoint:", id2)

    // diffWorking from first checkpoint should show the change
    const diff = await Effect.runPromise(vcs.diffWorking({ against: id1 }))
    console.log("Diff against initial:", JSON.stringify(diff, null, 2))
    expect(diff.files.length).toBe(1)
    expect(diff.files[0]!.path).toBe("test.txt")
  })

  test("subdirectory scenario - worktree is inside user's git repo", async () => {
    // Simulates running from a sub-directory of a real git repo
    const subDir = path.join(tmpDir, "packages", "foo")
    await fs.mkdir(subDir, { recursive: true })
    await fs.writeFile(path.join(subDir, "sub.txt"), "sub original")

    const gitDirPath = path.join(subDir, ".shadow", ".git")
    await fs.mkdir(gitDirPath, { recursive: true })

    const backend = await createJustGitBackend(subDir, gitDirPath, realFs)
    const vcs = await buildShadowVcs(backend, subDir)

    const id1 = await Effect.runPromise(vcs.record())
    console.log("Subdir checkpoint:", id1)

    await fs.writeFile(path.join(subDir, "sub.txt"), "sub modified")

    const changed = await Effect.runPromise(
      Effect.gen(function* () {
        return yield* vcs.changedSinceHead
      })
    )
    console.log("Subdir changed:", changed)
    expect(changed).toContain("sub.txt")
  })
})
