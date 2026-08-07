import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Option } from "effect"
import { describe, expect, it } from "vitest"
import { makeAcnOwnerLock } from "./owner-lock"
import { BunSqliteMutexLayer } from "./bun"

const fixtureAt = (name: string) =>
  resolve(fileURLToPath(new URL(".", import.meta.url)), `fixtures/${name}`)

const waitForFile = async (path: string): Promise<void> => {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (await Bun.file(path).exists()) return
    await Bun.sleep(10)
  }
  throw new Error(`Timed out waiting for ${path}`)
}

describe("ACN owner exclusion", () => {
  it("admits service initialization in at most one contending process", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-owner-exclusion-"))
    const barrier = join(root, "barrier")
    const admissions = join(root, "admissions")
    const fixture = fixtureAt("owner-contender.ts")
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

  it("releases ownership when the holder process is killed", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-owner-crash-"))
    const ready = join(root, "ready")
    const child = Bun.spawn([process.execPath, fixtureAt("owner-holder.ts"), root, ready], {
      stdin: "ignore",
      stdout: "ignore",
      stderr: "inherit",
    })
    try {
      await waitForFile(ready)
      await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
        const lock = yield* makeAcnOwnerLock(root)
        expect(Option.isNone(yield* lock.tryAcquire)).toBe(true)
      }).pipe(Effect.provide(Layer.merge(BunContext.layer, BunSqliteMutexLayer)))))

      child.kill(9)
      await child.exited

      await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
        const lock = yield* makeAcnOwnerLock(root)
        expect(Option.isSome(yield* lock.tryAcquire)).toBe(true)
      }).pipe(Effect.provide(Layer.merge(BunContext.layer, BunSqliteMutexLayer)))))
    } finally {
      try { child.kill(9) } catch { /* already exited */ }
      await rm(root, { recursive: true, force: true })
    }
  })
})
