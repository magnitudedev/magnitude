import { describe, test, expect, beforeEach, afterEach } from "vitest"
import * as fs from "node:fs/promises"
import * as path from "node:path"
import { createJustGitBackend, realFs } from "../src/backends/just-git"
import { buildShadowVcs } from "../src/layer"
import { runEffect, expectFailure, isOperationNotFound } from "./helpers/assertions"
import type { ShadowVcs } from "../src/index"

describe("Sandbox smoke test", () => {
  let tmpDir: string
  let workTree: string
  let service: ShadowVcs

  beforeEach(async () => {
    tmpDir = await fs.mkdtemp("/tmp/vcs-sandbox-smoke-")
    workTree = tmpDir
    const gitDirPath = path.join(tmpDir, ".vcs", ".git")
    await fs.mkdir(gitDirPath, { recursive: true })

    const backend = await createJustGitBackend(workTree, gitDirPath, realFs)

    // Run initial checkpoint before building ShadowVcs
    await runEffect(backend.buildCommit({ message: "initial checkpoint" }))

    service = await buildShadowVcs(backend, workTree)
  })

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true })
  })

  test("initial checkpoint captures worktree files", async () => {
    const head = await runEffect(service.head)
    expect(head.operationId).toBeTruthy()
    expect(head.filesChanged).toEqual([])
  })

  test("write file, record, verify head", async () => {
    await fs.writeFile(path.join(workTree, "foo.txt"), "Hello World\n")
    const opId = await runEffect(service.record({ message: "add foo" }))
    expect(opId).toBeTruthy()

    const head = await runEffect(service.head)
    expect(head.filesChanged).toContain("foo.txt")
  })

  test("undo removes file, redo restores it", async () => {
    await fs.writeFile(path.join(workTree, "bar.txt"), "test\n")
    await runEffect(service.record({ message: "add bar" }))

    // Undo
    const undoResult = await runEffect(service.undo())
    expect(undoResult.restoredOperationId).toBeTruthy()

    const existsAfterUndo = await (async () => {
      try { await fs.access(path.join(workTree, "bar.txt")); return true } catch { return false }
    })()
    expect(existsAfterUndo).toBe(false)

    // Redo
    const redoResult = await runEffect(service.redo())
    const existsAfterRedo = await (async () => {
      try { await fs.access(path.join(workTree, "bar.txt")); return true } catch { return false }
    })()
    expect(existsAfterRedo).toBe(true)

    const content = await fs.readFile(path.join(workTree, "bar.txt"), "utf-8")
    expect(content).toBe("test\n")
  })

  test("isClean reflects worktree state", async () => {
    const cleanBefore = await runEffect(service.isClean)
    expect(cleanBefore).toBe(true)

    await fs.writeFile(path.join(workTree, "new.txt"), "new\n")
    const cleanAfter = await runEffect(service.isClean)
    expect(cleanAfter).toBe(false)
  })
})
