/**
 * REPRO: listRefs("refs/checkpoints/") returns names with double slash.
 *
 * just-git's listRefs appends the prefix literally, producing
 * "refs/checkpoints//1" instead of "refs/checkpoints/1".
 *
 * We keep the defensive strip in ref-management.ts.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { Effect } from "effect"
import { createJustGitBackend } from "../src/backends/just-git"
import { realFs } from "../src/backends/just-git"
import { writeCheckpointRef, updateHead } from "../src/ref-management"
import { mkdir, rm, writeFile } from "node:fs/promises"
import * as path from "node:path"
import * as os from "node:os"

describe("REPRO: listRefs double-slash", () => {
  let projectDir: string
  let storagePath: string

  beforeEach(async () => {
    projectDir = os.tmpdir() + "/repro-listrefs-" + Date.now()
    storagePath = path.join(projectDir, ".magnitude", ".vcs")
    await mkdir(projectDir, { recursive: true })
    await mkdir(path.join(projectDir, "src"), { recursive: true })
    await writeFile(path.join(projectDir, "foo.txt"), "Hello World\n")
  })

  afterEach(async () => {
    await rm(projectDir, { recursive: true, force: true }).catch(() => {})
  })

  it("listRefs finds checkpoint refs after updateRef", async () => {
    const backend = await createJustGitBackend(projectDir, storagePath + "/.git", realFs)

    // Initial checkpoint using buildCommit
    const commit = await Effect.runPromise(backend.buildCommit({ message: "" }))
    await Effect.runPromise(writeCheckpointRef(backend, "1", commit))
    await Effect.runPromise(updateHead(backend, commit))

    // Check: listRefs with various prefixes
    const allRefs = await Effect.runPromise(backend.listRefs(""))
    console.log("listRefs(''):", allRefs)

    const cpRefs = await Effect.runPromise(backend.listRefs("refs/checkpoints/"))
    console.log("listRefs('refs/checkpoints/'):", cpRefs)

    const refsRefs = await Effect.runPromise(backend.listRefs("refs/"))
    console.log("listRefs('refs/'):", refsRefs)

    const headsRefs = await Effect.runPromise(backend.listRefs("refs/heads/"))
    console.log("listRefs('refs/heads/'):", headsRefs)

    // Also check readRef directly
    const cp1 = await Effect.runPromise(backend.readRef("refs/checkpoints/1"))
    console.log("readRef('refs/checkpoints/1'):", cp1)

    expect(cpRefs.length).toBeGreaterThan(0)
  })
})
