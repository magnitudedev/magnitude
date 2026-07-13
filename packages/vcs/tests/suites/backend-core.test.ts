import { describe, test, expect, beforeEach, afterEach } from "vitest"
import { Effect } from "effect"
import * as fs from "node:fs/promises"
import * as path from "node:path"
import { createJustGitBackend, realFs } from "../../src/backends/just-git"
import { runEffect, expectFailure } from "../helpers/assertions"
import type { VcsBackend } from "../../src/backend"
import { VcsBackendError } from "../../src/errors"
import { buildShadowVcs } from "../../src/layer"
import { writeCheckpointRef } from "../../src/ref-management"

describe("Backend Core Operations", () => {
  let tmpDir: string
  let workTree: string
  let backend: VcsBackend

  beforeEach(async () => {
    tmpDir = await fs.mkdtemp("/tmp/vcs-backend-core-")
    workTree = path.join(tmpDir, "worktree")
    await fs.mkdir(workTree, { recursive: true })
    const gitDirPath = path.join(tmpDir, ".vcs", ".git")
    backend = await createJustGitBackend(workTree, gitDirPath, realFs)

    // Run initial checkpoint
    const initHash = await runEffect(
      backend.buildCommit({ message: "initial checkpoint" }),
    )
    await runEffect(writeCheckpointRef(backend, "1", initHash))
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

  // ── getChangedFiles ──────────────────────────────────────────────────
  describe("getChangedFiles", () => {
    test("empty repo with files — all worktree files are added", async () => {
      await writeFile("foo.txt", "hello")
      const diffs = await runEffect(backend.getChangedFiles())
      expect(diffs).toHaveLength(1)
      expect(diffs[0]).toEqual({ path: "foo.txt", status: "added" })
    })

    test("after first commit, no changes — returns empty", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      expect(hash).toBeTruthy()

      const diffs = await runEffect(backend.getChangedFiles())
      expect(diffs).toEqual([])
    })

    test("file modified — status modified", async () => {
      await writeFile("foo.txt", "hello")
      await buildCommitFromChanges("init")

      await writeFile("foo.txt", "world")
      const diffs = await runEffect(backend.getChangedFiles())
      expect(diffs).toHaveLength(1)
      expect(diffs[0]).toEqual({ path: "foo.txt", status: "modified" })
    })

    test("file deleted from worktree — status deleted", async () => {
      await writeFile("foo.txt", "hello")
      await buildCommitFromChanges("init")

      await deleteFile("foo.txt")
      const diffs = await runEffect(backend.getChangedFiles())
      expect(diffs).toHaveLength(1)
      expect(diffs[0]).toEqual({ path: "foo.txt", status: "deleted" })
    })

    test("new file in worktree — status added", async () => {
      await writeFile("foo.txt", "hello")
      await buildCommitFromChanges("init")

      await writeFile("bar.txt", "new")
      const diffs = await runEffect(backend.getChangedFiles())
      expect(diffs).toHaveLength(1)
      expect(diffs[0]).toEqual({ path: "bar.txt", status: "added" })
    })

    test("unchanged file — not in result", async () => {
      await writeFile("foo.txt", "hello")
      await buildCommitFromChanges("init")

      const diffs = await runEffect(backend.getChangedFiles())
      expect(diffs).toEqual([])
    })

    test("empty file — works correctly", async () => {
      await writeFile("empty.txt", "")
      const diffs = await runEffect(backend.getChangedFiles())
      expect(diffs).toHaveLength(1)
      expect(diffs[0]).toEqual({ path: "empty.txt", status: "added" })
    })

    test("nested directory structure — all nested files detected", async () => {
      await writeFile("a/b/c.txt", "nested")
      await buildCommitFromChanges("init")
      await writeFile("a/b/d.txt", "nested2")
      const diffs = await runEffect(backend.getChangedFiles())
      expect(diffs).toHaveLength(1)
      expect(diffs[0]).toEqual({ path: "a/b/d.txt", status: "added" })
    })

    test("binary file — hash comparison works", async () => {
      const binary = new Uint8Array([0x00, 0x01, 0xff, 0xfe])
      await writeFile("bin.dat", binary)
      const diffs = await runEffect(backend.getChangedFiles())
      expect(diffs).toHaveLength(1)
      expect(diffs[0]).toEqual({ path: "bin.dat", status: "added" })
    })

    test("multibyte UTF-8 file — hash computed correctly", async () => {
      await writeFile("utf8.txt", "日本語")
      const diffs = await runEffect(backend.getChangedFiles())
      expect(diffs).toHaveLength(1)
      expect(diffs[0]).toEqual({ path: "utf8.txt", status: "added" })
    })
  })

  // ── buildCommit ──────────────────────────────────────────────────────
  describe("buildCommit", () => {
    test("no changes — still creates a new commit", async () => {
      await writeFile("foo.txt", "hello")
      await buildCommitFromChanges("first")

      // Second commit with no changes — buildCommit still creates a new commit
      const hash2 = await runEffect(backend.buildCommit({ message: "second" }))
      expect(hash2).toBeTruthy()
    })

    test("with files — creates commit and advances branch", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("add foo")
      expect(hash).toBeTruthy()

      const head = await runEffect(backend.readHead())
      expect(head.kind).toBe("symbolic")
      if (head.kind === "symbolic") {
        const hash2 = await runEffect(backend.readRef("HEAD"))
        expect(hash2).toBe(hash)
      }
    })

    test("file deletion — removes file from tree", async () => {
      await writeFile("foo.txt", "hello")
      await writeFile("bar.txt", "world")
      await buildCommitFromChanges("init")

      await deleteFile("foo.txt")
      const hash2 = await buildCommitFromChanges("delete foo")

      const history = await runEffect(backend.walkHistory({ start: hash2, limit: 1 }))
      const treeHash = history[0]!.tree
      const flat = await runEffect(backend.walkTree(treeHash))
      const paths = flat.map((e) => e.path)
      expect(paths).not.toContain("foo.txt")
      expect(paths).toContain("bar.txt")
    })

    test("nested paths — creates nested tree objects", async () => {
      await writeFile("a/b/c.txt", "nested")
      const hash = await buildCommitFromChanges("nested")

      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree
      const flat = await runEffect(backend.walkTree(treeHash))
      expect(flat).toHaveLength(1)
      expect(flat[0]!.path).toBe("a/b/c.txt")
    })
  })

  // ── readRef / updateRef / listRefs ───────────────────────────────────
  describe("readRef and updateRef", () => {
    test("create new direct ref — stored in refStore", async () => {
      await runEffect(backend.updateRef("refs/tags/v1", "abc123"))
      const value = await runEffect(backend.readRef("refs/tags/v1"))
      expect(value).toBe("abc123")
    })

    test("overwrite existing ref — previous value replaced", async () => {
      await runEffect(backend.updateRef("refs/tags/v1", "abc123"))
      await runEffect(backend.updateRef("refs/tags/v1", "def456"))
      const value = await runEffect(backend.readRef("refs/tags/v1"))
      expect(value).toBe("def456")
    })

    test("symbolic ref dereference — returns resolved hash", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")

      const head = await runEffect(backend.readRef("HEAD"))
      expect(head).toBe(hash)
    })

    test("read nonexistent ref — returns null", async () => {
      const value = await runEffect(backend.readRef("refs/tags/nonexistent"))
      expect(value).toBeNull()
    })

    test("readHead — returns symbolic ref after commit", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")

      const head = await runEffect(backend.readHead())
      expect(head.kind).toBe("symbolic")
      if (head.kind === "symbolic") {
        expect(head.target).toBe("refs/heads/main")
      }
    })
  })

  describe("listRefs", () => {
    test("empty prefix — returns checkpoint refs", async () => {
      const refs = await runEffect(backend.listRefs("refs/tags/"))
      expect(refs).toEqual([])
    })

    test("list checkpoint refs — returns checkpoint refs", async () => {
      const refs = await runEffect(backend.listRefs("refs/checkpoints/"))
      expect(refs.length).toBeGreaterThan(0)
      expect(refs[0]!.ref).toMatch(/^refs\/checkpoints\/\/?\d+$/)
      expect(refs[0]!.hash).toBeTruthy()
    })

    test("non-matching prefix — empty array", async () => {
      const refs = await runEffect(backend.listRefs("refs/tags/"))
      expect(refs).toEqual([])
    })
  })

  // ── diffTree ──────────────────────────────────────────────────────────
  describe("diffTree", () => {
    test("two identical trees — empty delta", async () => {
      await writeFile("foo.txt", "hello")
      const hash1 = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash1, limit: 1 }))
      const treeHash = history[0]!.tree

      const delta = await runEffect(backend.diffTree(treeHash, treeHash))
      expect(delta.additions).toBe(0)
      expect(delta.deletions).toBe(0)
      expect(delta.modifications).toBe(0)
      expect(delta.files).toEqual([])
    })

    test("from empty tree to tree with files — all added", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      const delta = await runEffect(backend.diffTree("", treeHash))
      expect(delta.additions).toBe(1)
      expect(delta.deletions).toBe(0)
      expect(delta.files[0]!.status).toBe("added")
    })

    test("from tree with files to empty tree — all deleted", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      const delta = await runEffect(backend.diffTree(treeHash, ""))
      expect(delta.additions).toBe(0)
      expect(delta.deletions).toBe(1)
      expect(delta.files[0]!.status).toBe("deleted")
    })

    test("file modified — shows modified", async () => {
      await writeFile("foo.txt", "hello")
      const hash1 = await buildCommitFromChanges("init")
      const history1 = await runEffect(backend.walkHistory({ start: hash1, limit: 1 }))
      const tree1 = history1[0]!.tree

      await writeFile("foo.txt", "world")
      const hash2 = await buildCommitFromChanges("modify")
      const history2 = await runEffect(backend.walkHistory({ start: hash2, limit: 1 }))
      const tree2 = history2[0]!.tree

      const delta = await runEffect(backend.diffTree(tree1, tree2))
      expect(delta.modifications).toBe(1)
      expect(delta.files[0]!.status).toBe("modified")
    })

    test("returns unfiltered tree diffs", async () => {
      await writeFile("src/a.ts", "a")
      await writeFile("src/b.ts", "b")
      await writeFile("test/c.ts", "c")
      const hash1 = await buildCommitFromChanges("init")
      const history1 = await runEffect(backend.walkHistory({ start: hash1, limit: 1 }))
      const tree1 = history1[0]!.tree

      await writeFile("src/a.ts", "a-modified")
      await writeFile("test/c.ts", "c-modified")
      const hash2 = await buildCommitFromChanges("modify")
      const history2 = await runEffect(backend.walkHistory({ start: hash2, limit: 1 }))
      const tree2 = history2[0]!.tree

      const delta = await runEffect(backend.diffTree(tree1, tree2))
      expect(delta.modifications).toBe(2)
      expect(delta.files.map((file) => file.path).sort()).toEqual(["src/a.ts", "test/c.ts"])
    })

    test("nonexistent from-tree — error", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      const error = await expectFailure(backend.diffTree("badhash", treeHash))
      expect(error).toBeInstanceOf(VcsBackendError)
    })

    test("nonexistent to-tree — error", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      const error = await expectFailure(backend.diffTree(treeHash, "badhash"))
      expect(error).toBeInstanceOf(VcsBackendError)
    })
  })

  // ── extractTree ──────────────────────────────────────────────────────
  describe("extractTree", () => {
    test("full restore to tree with fewer files — deletes untracked files", async () => {
      await writeFile("foo.txt", "hello")
      const hash1 = await buildCommitFromChanges("first")
      const history1 = await runEffect(backend.walkHistory({ start: hash1, limit: 1 }))
      const tree1 = history1[0]!.tree

      await writeFile("bar.txt", "bar")
      await buildCommitFromChanges("second")

      await runEffect(backend.extractTree(tree1, { kind: "full" }))

      expect(await fileExists("foo.txt")).toBe(true)
      expect(await fileExists("bar.txt")).toBe(false)
    })

    test("full restore to tree with files — writes files correctly", async () => {
      await writeFile("old.txt", "old")
      const hash1 = await buildCommitFromChanges("first")
      const history1 = await runEffect(backend.walkHistory({ start: hash1, limit: 1 }))
      const tree1 = history1[0]!.tree

      await deleteFile("old.txt")
      await writeFile("new.txt", "new")
      await buildCommitFromChanges("second")

      await runEffect(backend.extractTree(tree1, { kind: "full" }))

      expect(await fileExists("old.txt")).toBe(true)
      expect(await fileExists("new.txt")).toBe(false)
      expect(await readFile("old.txt")).toBe("old")
    })

    test("scoped restore (single file) — only that file written, others untouched", async () => {
      await writeFile("a.txt", "a")
      await writeFile("b.txt", "b")
      const hash1 = await buildCommitFromChanges("init")
      const history1 = await runEffect(backend.walkHistory({ start: hash1, limit: 1 }))
      const tree1 = history1[0]!.tree

      await writeFile("a.txt", "a-modified")
      await writeFile("b.txt", "b-modified")

      await runEffect(backend.extractTree(tree1, { kind: "file", path: "a.txt" }))
      expect(await readFile("a.txt")).toBe("a")
      expect(await readFile("b.txt")).toBe("b-modified")
    })

    test("scoped restore (directory) — only files under directory affected", async () => {
      await writeFile("src/a.ts", "a")
      await writeFile("src/b.ts", "b")
      await writeFile("test/c.ts", "c")
      const hash1 = await buildCommitFromChanges("init")
      const history1 = await runEffect(backend.walkHistory({ start: hash1, limit: 1 }))
      const tree1 = history1[0]!.tree

      await writeFile("src/a.ts", "a-modified")
      await writeFile("src/b.ts", "b-modified")
      await writeFile("test/c.ts", "c-modified")

      await runEffect(backend.extractTree(tree1, { kind: "directory", path: "src" }))
      expect(await readFile("src/a.ts")).toBe("a")
      expect(await readFile("src/b.ts")).toBe("b")
      expect(await readFile("test/c.ts")).toBe("c-modified")
    })

    test("restore file with nested path — creates parent directories", async () => {
      await writeFile("a/b/c.txt", "nested")
      const hash = await buildCommitFromChanges("nested")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      await deleteFile("a/b/c.txt")

      await runEffect(backend.extractTree(treeHash, { kind: "file", path: "a/b/c.txt" }))
      expect(await readFile("a/b/c.txt")).toBe("nested")
    })

    test("empty file in target tree — written as empty file", async () => {
      await writeFile("empty.txt", "")
      const hash = await buildCommitFromChanges("empty")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      await deleteFile("empty.txt")
      await runEffect(backend.extractTree(treeHash, { kind: "file", path: "empty.txt" }))
      expect(await readFile("empty.txt")).toBe("")
    })
  })

  // ── readFileAt ────────────────────────────────────────────────────────
  describe("readFileAt", () => {
    test("existing file — returns Uint8Array of content", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      const bytes = await runEffect(backend.readFileAt(treeHash, "foo.txt"))
      expect(bytes).toBeInstanceOf(Uint8Array)
      expect(new TextDecoder().decode(bytes!)).toBe("hello")
    })

    test("missing file — returns null", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      const bytes = await runEffect(backend.readFileAt(treeHash, "missing.txt"))
      expect(bytes).toBeNull()
    })

    test("empty file — returns empty Uint8Array", async () => {
      await writeFile("empty.txt", "")
      const hash = await buildCommitFromChanges("empty")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      const bytes = await runEffect(backend.readFileAt(treeHash, "empty.txt"))
      expect(bytes).toBeInstanceOf(Uint8Array)
      expect(bytes!.length).toBe(0)
    })

    test("binary file — returns raw bytes", async () => {
      const binary = new Uint8Array([0x00, 0x01, 0xff, 0xfe])
      await writeFile("bin.dat", binary)
      const hash = await buildCommitFromChanges("binary")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      const bytes = await runEffect(backend.readFileAt(treeHash, "bin.dat"))
      expect(bytes).toBeInstanceOf(Uint8Array)
      expect(bytes).not.toBeNull()
      expect(bytes!.length).toBeGreaterThan(0)
    })
  })

  // ── deleteRef ─────────────────────────────────────────────────────────
  describe("deleteRef", () => {
    test("delete existing ref — ref removed", async () => {
      await runEffect(backend.updateRef("refs/tags/v1", "abc123"))
      await runEffect(backend.deleteRef("refs/tags/v1"))
      const value = await runEffect(backend.readRef("refs/tags/v1"))
      expect(value).toBeNull()
    })

    test("delete nonexistent ref — silent no-op", async () => {
      await runEffect(backend.deleteRef("refs/tags/nonexistent"))
      const value = await runEffect(backend.readRef("refs/tags/nonexistent"))
      expect(value).toBeNull()
    })
  })

  // ── walkHistory ──────────────────────────────────────────────────────
  describe("walkHistory", () => {
    test("from HEAD — walks from HEAD backwards", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")

      const history = await runEffect(backend.walkHistory())
      // 2 commits: initial checkpoint + "init" we just created
      expect(history).toHaveLength(2)
      expect(history[0]!.hash).toBe(hash)
    })

    test("with limit — returns at most N commits", async () => {
      await writeFile("a.txt", "1")
      await buildCommitFromChanges("c1")

      await writeFile("a.txt", "2")
      await buildCommitFromChanges("c2")

      await writeFile("a.txt", "3")
      await buildCommitFromChanges("c3")

      const history = await runEffect(backend.walkHistory({ limit: 2 }))
      expect(history).toHaveLength(2)
    })

    test("start from specific commit hash", async () => {
      await writeFile("a.txt", "1")
      const c1 = await buildCommitFromChanges("c1")

      const history = await runEffect(backend.walkHistory({ start: c1, limit: 1 }))
      expect(history).toHaveLength(1)
      expect(history[0]!.hash).toBe(c1)
    })

    test("pathFilter — only commits touching that path", async () => {
      await writeFile("a.txt", "1")
      const c1 = await buildCommitFromChanges("c1")

      await writeFile("b.txt", "1")
      await buildCommitFromChanges("c2")

      const history = await runEffect(backend.walkHistory({ pathFilter: "a.txt" }))
      expect(history).toHaveLength(1)
      expect(history[0]!.hash).toBe(c1)
    })
  })

  // ── walkTree ─────────────────────────────────────────────────────────
  describe("walkTree", () => {
    test("basic — returns all entries with path, hash, mode", async () => {
      await writeFile("foo.txt", "hello")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      const flat = await runEffect(backend.walkTree(treeHash))
      expect(flat).toHaveLength(1)
      expect(flat[0]!.path).toBe("foo.txt")
      expect(flat[0]!.hash).toBeTruthy()
      expect(flat[0]!.mode).toBeTruthy()
    })

    test("nested paths — returns all entries flattened", async () => {
      await writeFile("a/b/c.txt", "nested")
      const hash = await buildCommitFromChanges("nested")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      const flat = await runEffect(backend.walkTree(treeHash))
      expect(flat).toHaveLength(1)
      expect(flat[0]!.path).toBe("a/b/c.txt")
    })

    test("multiple files — sorted", async () => {
      await writeFile("b.txt", "b")
      await writeFile("a.txt", "a")
      const hash = await buildCommitFromChanges("init")
      const history = await runEffect(backend.walkHistory({ start: hash, limit: 1 }))
      const treeHash = history[0]!.tree

      const flat = await runEffect(backend.walkTree(treeHash))
      const paths = flat.map((e) => e.path)
      expect(paths).toEqual(["a.txt", "b.txt"])
    })
  })
})
