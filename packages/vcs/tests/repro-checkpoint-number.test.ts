/**
 * REPRO: nextCheckpointNumber / writeCheckpointRef bug
 *
 * record() creates a new commit and updates HEAD, but checkpoint refs
 * don't increase. Only 1 checkpoint visible after multiple records.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { Effect } from "effect"
import { createJustGitBackend } from "../src/backends/just-git"
import { realFs } from "../src/backends/just-git"
import { writeCheckpointRef, updateHead, nextCheckpointNumber, listCheckpointRefs, readHead } from "../src/ref-management"
import { mkdir, rm, writeFile } from "node:fs/promises"
import * as path from "node:path"
import * as os from "node:os"

describe("REPRO: checkpoint numbering bug", () => {
  let projectDir: string
  let storagePath: string

  beforeEach(async () => {
    projectDir = os.tmpdir() + "/repro-cpnum-" + Date.now()
    storagePath = path.join(projectDir, ".magnitude", ".vcs")
    await mkdir(projectDir, { recursive: true })
    await mkdir(path.join(projectDir, "src"), { recursive: true })
    await writeFile(path.join(projectDir, "foo.txt"), "Hello World\n")
  })

  afterEach(async () => {
    await rm(projectDir, { recursive: true, force: true }).catch(() => {})
  })

  it("nextCheckpointNumber increments after writing a checkpoint ref", async () => {
    const backend = await createJustGitBackend(projectDir, storagePath + "/.git", realFs)

    // Initial checkpoint
    const commit1 = await Effect.runPromise(backend.buildCommit({ message: "" }))
    await Effect.runPromise(writeCheckpointRef(backend, "1", commit1))
    await Effect.runPromise(updateHead(backend, commit1))

    // Check nextCheckpointNumber
    const next1 = await Effect.runPromise(nextCheckpointNumber(backend))
    console.log("nextCheckpointNumber after initial:", next1)
    expect(next1).toBe(2)

    // List refs
    const refs1 = await Effect.runPromise(listCheckpointRefs(backend))
    console.log("Refs after initial:", refs1)

    // Now write checkpoint 2
    const head = await Effect.runPromise(readHead(backend))
    console.log("HEAD:", head?.slice(0, 7))

    await Effect.runPromise(writeCheckpointRef(backend, "2", head!))

    const next2 = await Effect.runPromise(nextCheckpointNumber(backend))
    console.log("nextCheckpointNumber after writing 2:", next2)
    expect(next2).toBe(3)

    const refs2 = await Effect.runPromise(listCheckpointRefs(backend))
    console.log("Refs after writing 2:", refs2)
    expect(refs2.length).toBe(2)
  })

  it("simulates what record() does step by step", async () => {
    const backend = await createJustGitBackend(projectDir, storagePath + "/.git", realFs)

    // Initial checkpoint (same as makeShadowVcs)
    const commit1 = await Effect.runPromise(backend.buildCommit({ message: "" }))
    await Effect.runPromise(writeCheckpointRef(backend, "1", commit1))
    await Effect.runPromise(updateHead(backend, commit1))
    console.log("Initial commit:", commit1.slice(0, 7))

    // Modify file
    await writeFile(path.join(projectDir, "foo.txt"), "Modified!\n")

    // Simulate record() step by step using new interface
    const result = await Effect.runPromise(
      Effect.gen(function* () {
        // Step 1: getChangedFiles
        const changes = yield* backend.getChangedFiles()
        console.log("Changes:", changes.map(c => ({ path: c.path, status: c.status })))

        // Step 2: buildCommit (index handles staging)
        const commitHash = yield* backend.buildCommit({ message: "modified" })
        console.log("New commit:", commitHash.slice(0, 7))

        // Step 4: nextCheckpointNumber
        const nextNum = yield* nextCheckpointNumber(backend)
        console.log("nextCheckpointNumber:", nextNum)

        // Step 5: writeCheckpointRef
        yield* writeCheckpointRef(backend, String(nextNum), commitHash)
        console.log("Wrote checkpoint ref:", String(nextNum))

        // Step 6: updateHead
        yield* updateHead(backend, commitHash)

        // Verify
        const refs = yield* listCheckpointRefs(backend)
        console.log("All checkpoint refs:", refs)
        return refs
      }),
    )

    expect(result.length).toBe(2)
  })
})
