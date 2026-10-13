import { describe, it, expect, afterEach } from "vitest"
import { Effect, Layer } from "effect"
import * as path from "node:path"
import * as os from "node:os"
import * as fs from "node:fs/promises"
import { ShadowVcs, makeShadowVcsLayer, VcsFs } from "../src/index"
import { realFs } from "../src/backends/just-git"

const umasks = [["002", 0o002], ["022", 0o022]] as const
const posix = process.platform !== "win32"

describe.skipIf(!posix).each(umasks)("checkpoints under umask %s", (_label, umask) => {
  let previous: number | undefined
  let tmpDir: string | undefined

  afterEach(async () => {
    if (previous !== undefined) process.umask(previous)
    previous = undefined
    if (tmpDir) await fs.rm(tmpDir, { recursive: true, force: true })
    tmpDir = undefined
  })

  it("records git tree modes and restores files", async () => {
    previous = process.umask(umask)
    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "vcs-umask-"))
    const worktree = tmpDir

    await fs.writeFile(path.join(worktree, "README.md"), "readme\n")
    await fs.mkdir(path.join(worktree, "bin"))
    await fs.writeFile(path.join(worktree, "bin", "run.sh"), "#!/bin/sh\n", { mode: 0o777 })
    await fs.symlink("README.md", path.join(worktree, "link.md"))

    const created = await fs.stat(path.join(worktree, "README.md"))
    expect(created.mode & 0o777).toBe(0o666 & ~umask)

    const layer = makeShadowVcsLayer({ worktreePath: worktree, storagePath: path.join(worktree, ".vcs") }).pipe(
      Layer.provide(Layer.succeed(VcsFs, realFs)),
    )
    const vcs = await Effect.runPromise(ShadowVcs.pipe(Effect.provide(layer)))
    const run = <A, E>(effect: Effect.Effect<A, E>) => Effect.runPromise(effect)

    const first = await run(vcs.record({ message: "initial" }))
    expect(first).toBeTruthy()

    expect((await realFs.stat(path.join(worktree, "README.md"))).mode).toBe(0o100644)
    expect((await realFs.stat(path.join(worktree, "bin", "run.sh"))).mode).toBe(0o100755)
    expect((await realFs.lstat(path.join(worktree, "link.md"))).mode).toBe(0o120000)
    expect((await realFs.stat(path.join(worktree, "bin"))).mode).toBe(0o040000)

    await fs.writeFile(path.join(worktree, "README.md"), "changed\n")
    await fs.writeFile(path.join(worktree, "notes.txt"), "new\n")
    const second = await run(vcs.record({ message: "edit" }))
    expect(second).toBeTruthy()
    expect(second).not.toBe(first)

    await run(vcs.undo())
    expect(await fs.readFile(path.join(worktree, "README.md"), "utf8")).toBe("readme\n")
    await expect(fs.access(path.join(worktree, "notes.txt"))).rejects.toThrow()
    expect(await fs.readlink(path.join(worktree, "link.md"))).toBe("README.md")
  })
})
