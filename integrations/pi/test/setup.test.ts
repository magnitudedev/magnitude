import { Effect, Schema } from "effect"
import { FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent"
import { describe, expect, it, vi } from "vitest"
import { openMagnitudeDesktop, registerMagnitudeSetup } from "../extensions/setup"

const encode = Schema.encodeSync(Schema.parseJson(Schema.String))
const fixture = (code: number) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = yield* fs.makeTempDirectoryScoped({ prefix: "pi-desktop-" })
  const executable = `${directory}/magnitude`
  yield* fs.writeFileString(executable, `#!${process.execPath}\nif (process.argv.slice(2).join(' ') !== 'app open') process.exit(99)\nrequire('node:fs').writeFileSync(${encode(`${directory}/opened`)}, 'yes')\nif (${code}) console.error('Desktop is not installed')\nprocess.exit(${code})\n`)
  yield* fs.chmod(executable, 0o755)
  const previous = process.env.MAGNITUDE_CLI
  process.env.MAGNITUDE_CLI = executable
  yield* Effect.addFinalizer(() => Effect.sync(() => {
    if (previous === undefined) delete process.env.MAGNITUDE_CLI
    else process.env.MAGNITUDE_CLI = previous
  }))
  return { directory, fs }
})
const run = <A, E>(effect: Effect.Effect<A, E, import("effect").Scope.Scope | import("@effect/platform/FileSystem").FileSystem | import("@effect/platform/CommandExecutor").CommandExecutor>) => Effect.runPromise(Effect.scoped(effect).pipe(Effect.provide(NodeContext.layer)))

describe("Pi desktop setup handoff", () => {
  it.each([0, 1])("runs the actual headless navigation command with exit %s", code => run(Effect.gen(function* () {
    const f = yield* fixture(code)
    const result = yield* openMagnitudeDesktop(f.directory).pipe(Effect.either)
    expect(result._tag).toBe(code === 0 ? "Right" : "Left")
    expect(yield* f.fs.readFileString(`${f.directory}/opened`)).toBe("yes")
    if (result._tag === "Left") expect(result.left.message).toContain("Desktop is not installed")
  })))
  it("opens without taking terminal ownership, selecting a model, or reloading Pi", () => run(Effect.gen(function* () {
    const f = yield* fixture(0)
    const registerCommand = vi.fn()
    const setModel = vi.fn()
    const custom = vi.fn()
    const reload = vi.fn()
    const notify = vi.fn()
    const setup = registerMagnitudeSetup({ registerCommand, setModel } as unknown as ExtensionAPI)
    yield* Effect.addFinalizer(() => Effect.promise(setup.dispose))
    const ctx = { cwd: f.directory, mode: "tui", isIdle: () => true, hasPendingMessages: () => false, reload, ui: { custom, notify } } as unknown as ExtensionContext
    expect(yield* Effect.promise(() => setup.run(ctx))).toBe(true)
    expect(notify).toHaveBeenCalledWith(expect.stringContaining("connect Pi in Connections"), "info")
    expect(setModel).not.toHaveBeenCalled()
    expect(custom).not.toHaveBeenCalled()
    expect(reload).not.toHaveBeenCalled()
  })))
  it.each(["headless", "busy"])("does not open from %s context", condition => run(Effect.gen(function* () {
    const f = yield* fixture(0)
    const notify = vi.fn()
    const setup = registerMagnitudeSetup({ registerCommand: vi.fn() } as unknown as ExtensionAPI)
    yield* Effect.addFinalizer(() => Effect.promise(setup.dispose))
    const ctx = { cwd: f.directory, mode: condition === "headless" ? "rpc" : "tui", isIdle: () => condition !== "busy", hasPendingMessages: () => false, ui: { notify } } as unknown as ExtensionContext
    expect(yield* Effect.promise(() => setup.run(ctx))).toBe(false)
    expect(yield* f.fs.exists(`${f.directory}/opened`)).toBe(false)
    expect(notify).toHaveBeenCalledWith(expect.any(String), "error")
  })))
})
