import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { BunDetachedSpawnProcess } from "./bun-spawn-process"

describe("BunDetachedSpawnProcess", () => {
  it("drains stdout and stderr into a bounded diagnostic tail", async () => {
    const spawned = await Effect.runPromise(
      BunDetachedSpawnProcess.spawn([
        globalThis.process.execPath,
        "-e",
        [
          `process.stdout.write("x".repeat(${70 * 1024}), () => {`,
          `  process.stderr.write("candidate failed\\n", () => process.exit(7))`,
          "})",
        ].join(";"),
      ]),
    )

    expect(await Effect.runPromise(spawned.exited)).toBe(7)
    const diagnostic = await Effect.runPromise(spawned.diagnostic)
    expect(Option.isSome(diagnostic)).toBe(true)
    if (Option.isSome(diagnostic)) {
      expect(new TextEncoder().encode(diagnostic.value).byteLength).toBeLessThanOrEqual(
        64 * 1024,
      )
      expect(diagnostic.value).toContain("candidate failed")
    }
  })
})
