import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { BunDetachedChildProcessSpawner } from "./bun-spawn-process"

describe("BunDetachedChildProcessSpawner", () => {
  it("returns a scope-owned candidate with a mandatory PID", async () => {
    await Effect.runPromise(
      Effect.scoped(
        Effect.gen(function* () {
          const spawned = yield* BunDetachedChildProcessSpawner.spawn([
            globalThis.process.execPath,
            "-e",
            [
              `process.stdout.write("x".repeat(${70 * 1024}), () => {`,
              `  process.stderr.write("candidate failed\\n", () => process.exit(7))`,
              "})",
            ].join(";"),
          ])
          expect(spawned.pid).toBeGreaterThan(0)
        }),
      ),
    )
  })
})
