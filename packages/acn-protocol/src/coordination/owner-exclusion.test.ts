import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const fixture = resolve(
  fileURLToPath(new URL(".", import.meta.url)),
  "fixtures/owner-contender.ts",
)

describe("ACN owner admission", () => {
  it("admits service initialization in exactly one contending process", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-owner-admission-"))
    const barrier = join(root, "barrier")
    const admissions = join(root, "admissions")
    const spawn = () => Bun.spawn([process.execPath, fixture, root, barrier, admissions], {
      stdin: "ignore",
      stdout: "ignore",
      stderr: "inherit",
    })
    const first = spawn()
    const second = spawn()
    try {
      await Bun.sleep(25)
      await writeFile(barrier, "go")
      expect(await Promise.all([first.exited, second.exited])).toEqual([0, 0])
      const lines = (await readFile(admissions, "utf8")).trim().split("\n").filter(Boolean)
      expect(lines).toHaveLength(1)
    } finally {
      try { first.kill(9) } catch { /* already exited */ }
      try { second.kill(9) } catch { /* already exited */ }
      await rm(root, { recursive: true, force: true })
    }
  })
})
