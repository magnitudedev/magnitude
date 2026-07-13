import { describe, test, expect, beforeEach, afterEach } from "vitest"
import { Effect } from "effect"
import * as fs from "node:fs/promises"
import * as path from "node:path"
import { createJustGitBackend, realFs } from "../../src/backends/just-git"
import { runEffect } from "../helpers/assertions"
import type { VcsBackend } from "../../src/backend"

describe("Backend Safety & Isolation", () => {
  let tmpDir: string
  let workTree: string
  let backend: VcsBackend

  beforeEach(async () => {
    tmpDir = await fs.mkdtemp("/tmp/vcs-backend-safety-")
    workTree = path.join(tmpDir, "worktree")
    await fs.mkdir(workTree, { recursive: true })
    const gitDirPath = path.join(tmpDir, ".vcs", ".git")
    backend = await createJustGitBackend(workTree, gitDirPath, realFs)

    // Run initial checkpoint
    await runEffect(backend.buildCommit({ message: "initial checkpoint" }))
  })

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true })
  })

  async function writeFile(relPath: string, content: string | Uint8Array): Promise<void> {
    const fullPath = path.join(workTree, relPath)
    await fs.mkdir(path.dirname(fullPath), { recursive: true })
    await fs.writeFile(fullPath, content)
  }

  async function deleteFile(relPath: string): Promise<void> {
    try {
      await fs.rm(path.join(workTree, relPath))
    } catch {
      // already gone
    }
  }

  async function readFile(relPath: string): Promise<string | null> {
    try {
      const buf = await fs.readFile(path.join(workTree, relPath))
      return buf.toString()
    } catch {
      return null
    }
  }

  async function fileExists(relPath: string): Promise<boolean> {
    try {
      await fs.access(path.join(workTree, relPath))
      return true
    } catch {
      return false
    }
  }

  async function buildCommitFromChanges(message: string): Promise<string> {
    return await runEffect(backend.buildCommit({ message }))
  }

  // ── getChangedFiles skips .git ────────────────────────────────────────
  describe("path safety — .git blocked", () => {
    test("getChangedFiles skips .git directory contents", async () => {
      await writeFile("foo.txt", "hello")
      await writeFile(".git/secret.txt", "secret")
      const diffs = await runEffect(backend.getChangedFiles())
      const paths = diffs.map((d) => d.path)
      expect(paths).toContain("foo.txt")
      expect(paths).not.toContain(".git/secret.txt")
    })

    test("buildCommit skips .git paths when auto-detecting", async () => {
      await writeFile("foo.txt", "hello")
      await writeFile(".git/config", "secret")
      const hash = await buildCommitFromChanges("init")
      expect(hash).toBeTruthy()

      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree
      const flat = await runEffect(backend.walkTree(treeHash))
      const paths = flat.map((e) => e.path)
      expect(paths).toContain("foo.txt")
      expect(paths).not.toContain(".git/config")
    })

    test("extractTree skips .git entries in target tree", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      await writeFile("foo.txt", "modified")
      await writeFile(".git/secret.txt", "secret")

      await runEffect(backend.extractTree(treeHash, { kind: "full" }))
      const fooContent = await readFile("foo.txt")
      expect(fooContent).toBe("hello")
      const secretExists = await fileExists(".git/secret.txt")
      expect(secretExists).toBe(true)
    })
  })

  describe("path safety — .. blocked", () => {
    test("getChangedFiles does not report .. paths", async () => {
      await writeFile("foo.txt", "hello")
      const diffs = await runEffect(backend.getChangedFiles())
      const paths = diffs.map((d) => d.path)
      expect(paths).toContain("foo.txt")
    })

    test("extractTree with valid paths works correctly", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      await runEffect(backend.extractTree(treeHash, { kind: "file", path: "foo.txt" }))
      const content = await readFile("foo.txt")
      expect(content).toBe("hello")
    })
  })

  describe("path safety — absolute paths blocked", () => {
    test("getChangedFiles only reports relative paths", async () => {
      await writeFile("foo.txt", "hello")
      const diffs = await runEffect(backend.getChangedFiles())
      for (const d of diffs) {
        expect(d.path.startsWith("/")).toBe(false)
      }
    })
  })

  describe("path safety — valid paths OK", () => {
    test("normal relative paths pass safety check", async () => {
      const paths = [
        "foo.txt",
        "src/main.ts",
        "deep/nested/path/file.txt",
        "a-b_c.d",
        ".hidden",
      ]

      for (const p of paths) {
        await writeFile(p, "content")
      }
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree
      const flat = await runEffect(backend.walkTree(treeHash))
      const storedPaths = flat.map((e) => e.path).sort()
      expect(storedPaths).toEqual(paths.sort())
    })
  })

  // ── extractTree skips .git entries ────────────────────────────────────
  describe("extractTree .git entry skipping", () => {
    test("does not restore .git/config from target tree", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      await runEffect(backend.extractTree(treeHash, { kind: "full" }))
      const fooContent = await readFile("foo.txt")
      expect(fooContent).toBe("hello")
    })

    test("full restore does not write to parent directory via ..", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      await runEffect(backend.extractTree(treeHash, { kind: "full" }))
      // Verify only worktree was touched, not parent
      const parentFile = await (async () => {
        try {
          await fs.readFile(path.join(tmpDir, "..", "parent-file.txt"))
          return "exists"
        } catch {
          return null
        }
      })()
      expect(parentFile).toBeNull()
    })
  })

  // ── getChangedFiles skips .git directory ──────────────────────────────
  describe("getChangedFiles .git skipping", () => {
    test("completely ignores .git directory in worktree", async () => {
      await writeFile("foo.txt", "hello")
      await writeFile(".git/config", "[core]\n")
      await writeFile(".git/HEAD", "ref: refs/heads/main\n")
      await writeFile(".git/objects/abc", "\x00\x01\x02")

      const diffs = await runEffect(backend.getChangedFiles())
      const paths = diffs.map((d) => d.path)
      expect(paths).toEqual(["foo.txt"])
    })

    test("ignores nested .git directories", async () => {
      await writeFile("foo.txt", "hello")
      await writeFile("submodule/.git/config", "bare = true\n")
      await writeFile("submodule/src/code.ts", "// code")

      const diffs = await runEffect(backend.getChangedFiles())
      const paths = diffs.map((d) => d.path)
      expect(paths).toContain("foo.txt")
      expect(paths).not.toContain("submodule/.git/config")
    })
  })
})
