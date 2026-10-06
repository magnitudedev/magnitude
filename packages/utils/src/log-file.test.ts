import { describe, expect, test } from "vitest"
import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { mkdtempSync, readFileSync, existsSync, writeFileSync, chmodSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { openLogFile } from "./log-file"

const run = <A>(effect: Effect.Effect<A, unknown, any>) =>
  Effect.runPromise(effect.pipe(Effect.scoped, Effect.provide(NodeContext.layer)) as Effect.Effect<A>)

const directory = () => mkdtempSync(join(tmpdir(), "log-file-"))

describe("openLogFile", () => {
  test("appends to an existing file and writes queued output when the scope closes", async () => {
    const file = join(directory(), "logs", "service.log")
    await run(Effect.gen(function* () {
      const log = yield* openLogFile(file, 1024)
      yield* log.append("first\n")
      yield* log.append(new TextEncoder().encode("second\n"))
    }))
    await run(Effect.gen(function* () {
      const log = yield* openLogFile(file, 1024)
      yield* log.append("third\n")
    }))
    expect(readFileSync(file, "utf8")).toBe("first\nsecond\nthird\n")
  })

  test("rotates once to .1 when the next write would exceed the bound", async () => {
    const file = join(directory(), "service.log")
    await run(Effect.gen(function* () {
      const log = yield* openLogFile(file, 10)
      yield* log.append("aaaaaaaa\n")
      yield* log.append("bbbbbbbb\n")
      yield* log.append("cccccccc\n")
    }))
    expect(readFileSync(file, "utf8")).toBe("cccccccc\n")
    expect(readFileSync(`${file}.1`, "utf8")).toBe("bbbbbbbb\n")
  })

  test("rotates a file already over the bound when it opens", async () => {
    const file = join(directory(), "service.log")
    writeFileSync(file, "x".repeat(20))
    writeFileSync(`${file}.1`, "older")
    await run(Effect.gen(function* () {
      const log = yield* openLogFile(file, 10)
      yield* log.append("new\n")
    }))
    expect(readFileSync(file, "utf8")).toBe("new\n")
    expect(readFileSync(`${file}.1`, "utf8")).toBe("x".repeat(20))
  })

  test.skipIf(process.platform === "win32" || process.getuid?.() === 0)("an unwritable location is skipped without failing", async () => {
    const parent = directory()
    chmodSync(parent, 0o500)
    const file = join(parent, "logs", "service.log")
    await run(Effect.gen(function* () {
      const log = yield* openLogFile(file, 1024)
      yield* log.append("ignored\n")
    }))
    expect(existsSync(file)).toBe(false)
  })
})
