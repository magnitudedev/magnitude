import { Deferred, Effect, Fiber, Layer, Schema } from "effect"
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent"
import { describe, expect, it, vi } from "vitest"
import { ModelIdSchema } from "@magnitudedev/sdk"
import { FileSystem } from "@effect/platform"
import * as Command from "@effect/platform/Command"
import { NodeContext } from "@effect/platform-node"
import { PiSetup, PiSetupFailed, PiSetupLive, prepareMagnitudeCli, registerMagnitudeSetup, validateSetupTermination, withPiTerminal, withPiPreparation } from "../extensions/setup"
import type { CancellableLoader } from "@earendil-works/pi-tui"

const modelId = ModelIdSchema.make("test:gguf:q4")
const completed = { _tag: "Completed", protocolVersion: 1, modelId } as const
const cancelled = { _tag: "Cancelled", protocolVersion: 1 } as const
const failed = { _tag: "Failed", protocolVersion: 1, message: "installation failed" } as const
const encodeString = Schema.encodeSync(Schema.parseJson(Schema.String))

describe("actual hosted child boundary", () => {
  it.each(["completed", "cancelled", "failed", "missing", "malformed", "oversized", "contradictory", "signal", "old-cli"])("handles %s and cleans the result directory", async scenario => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "pi-host-fixture-" })
      const executable = `${directory}/magnitude`
      const receipt = `${directory}/receipt`
      yield* fs.writeFileString(executable, `#!/usr/bin/env node
import { writeFileSync } from 'node:fs'
if (process.argv.includes('--host-protocol')) {
  console.log(${scenario === "old-cli" ? "'{\"protocolVersion\":0}'" : "'{\"protocolVersion\":1}'"})
} else {
  const path = process.argv[process.argv.indexOf('--result-file') + 1]
  writeFileSync(${encodeString(receipt)}, path)
  const scenario = ${encodeString(scenario)}
  if (scenario === 'signal') process.kill(process.pid, 'SIGKILL')
  if (scenario !== 'missing') writeFileSync(path, scenario === 'malformed' ? '{' : scenario === 'oversized' ? 'x'.repeat(17000) : JSON.stringify(
    scenario === 'failed' ? { _tag: 'Failed', protocolVersion: 1, message: 'fixture failure' } :
    scenario === 'cancelled' ? { _tag: 'Cancelled', protocolVersion: 1 } :
    { _tag: 'Completed', protocolVersion: 1, modelId: 'test:gguf:q4' }
  ))
  process.exitCode = scenario === 'failed' || scenario === 'contradictory' ? 1 : 0
}
`)
      yield* fs.chmod(executable, 0o755)
      const previous = process.env.MAGNITUDE_CLI
      yield* Effect.addFinalizer(() => Effect.sync(() => {
        if (previous === undefined) delete process.env.MAGNITUDE_CLI
        else process.env.MAGNITUDE_CLI = previous
      }))
      process.env.MAGNITUDE_CLI = executable
      const calls: string[] = []
      const ctx = { cwd: directory, ui: { custom: (factory: Function) => new Promise<void>(resolve => {
        factory({ terminal: { write: () => calls.push("reset") }, stop: () => calls.push("stop"), start: () => calls.push("start"), requestRender: () => {} }, { fg: (_: string, text: string) => text }, {}, resolve)
      }) } } as unknown as ExtensionContext
      const result = yield* Effect.flatMap(PiSetup, setup => setup.run(ctx)).pipe(
        Effect.provide(PiSetupLive), Effect.either,
      )
      if (scenario === "completed" || scenario === "cancelled") {
        expect(result).toMatchObject({ _tag: "Right", right: scenario === "completed" ? completed : cancelled })
      } else {
        expect(result._tag).toBe("Left")
      }
      if (scenario === "old-cli") {
        expect(calls).toEqual([])
        expect(yield* fs.exists(receipt)).toBe(false)
      } else {
        expect(calls).toEqual(["stop", "reset", "start"])
        const resultPath = yield* fs.readFileString(receipt)
        expect(yield* fs.exists(resultPath)).toBe(false)
      }
    })).pipe(Effect.provide(NodeContext.layer)))
  })
})

describe("first-encounter CLI installation", () => {
  it.each(["missing", "existing", "incompatible", "nonzero", "override", "permission", "npm-failed", "npm-missing", "interrupted", "bad-install"])("handles %s without an ambient installation", async scenario => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "pi-cli-bootstrap-" })
      const executable = `${directory}/magnitude`
      const receipt = `${directory}/install.json`
      const cli = `#!${process.execPath}\nconsole.log('{"protocolVersion":1}')\n`
      const previousPath = process.env.PATH
      const previousOverride = process.env.MAGNITUDE_CLI
      yield* Effect.addFinalizer(() => Effect.sync(() => {
        if (previousPath === undefined) delete process.env.PATH
        else process.env.PATH = previousPath
        if (previousOverride === undefined) delete process.env.MAGNITUDE_CLI
        else process.env.MAGNITUDE_CLI = previousOverride
      }))
      process.env.PATH = directory
      delete process.env.MAGNITUDE_CLI
      if (scenario === "override") process.env.MAGNITUDE_CLI = `${directory}/explicit-missing`
      if (["existing", "incompatible", "nonzero", "permission"].includes(scenario)) {
        yield* fs.writeFileString(executable, scenario === "incompatible" ? `#!${process.execPath}\nconsole.log('{}')\n` : cli + (scenario === "nonzero" ? "process.exitCode = 1\n" : ""))
        yield* fs.chmod(executable, scenario === "permission" ? 0o644 : 0o755)
      }
      if (scenario !== "npm-missing") {
        yield* fs.writeFileString(`${directory}/npm`, `#!${process.execPath}
const fs = require('node:fs')
fs.writeFileSync(${encodeString(receipt)}, JSON.stringify(process.argv.slice(2)))
console.log('hidden npm output')
if (${encodeString(scenario)} === 'npm-failed') { console.error('EACCES: test prefix not writable'); process.exit(1) }
if (${encodeString(scenario)} === 'interrupted') process.kill(process.pid, 'SIGTERM')
fs.writeFileSync(${encodeString(executable)}, ${encodeString(scenario === "bad-install" ? `#!${process.execPath}\nconsole.log('{}')\n` : cli)}, { mode: 0o755 })
`)
        yield* fs.chmod(`${directory}/npm`, 0o755)
      }
      const messages: string[] = []
      const result = yield* prepareMagnitudeCli(directory, message => Effect.sync(() => { messages.push(message) })).pipe(Effect.either)
      if (scenario === "missing") expect(messages).toEqual(["Installing Magnitude…"])
      if (scenario === "existing") expect(messages).toEqual([])
      if (scenario === "npm-failed") {
        expect(result).toMatchObject({ _tag: "Left", left: { message: expect.stringContaining("EACCES: test prefix not writable") } })
        expect(messages).toEqual(["Installing Magnitude…"])
      }
      if (scenario === "missing" || scenario === "existing") {
        expect(result).toMatchObject({ _tag: "Right", right: "magnitude" })
        // Re-entry must reuse the newly installed CLI, not install again.
        expect(yield* prepareMagnitudeCli(directory)).toBe("magnitude")
      } else expect(result._tag).toBe("Left")
      if (["missing", "npm-failed", "interrupted", "bad-install"].includes(scenario)) {
        expect(yield* Schema.decodeUnknown(Schema.parseJson(Schema.Array(Schema.String)))(yield* fs.readFileString(receipt))).toEqual(["install", "--global", "@magnitudedev/cli"])
      } else expect(yield* fs.exists(receipt)).toBe(false)
      if (scenario === "npm-failed") {
        yield* fs.writeFileString(`${directory}/npm`, `#!${process.execPath}\nrequire('node:fs').writeFileSync(${encodeString(executable)}, ${encodeString(cli)}, { mode: 0o755 })\n`)
        expect(yield* prepareMagnitudeCli(directory)).toBe("magnitude")
      }
    })).pipe(Effect.provide(NodeContext.layer)))
  })
})

describe("Pi preparation spinner", () => {
  it("reaps a real background child before closing the cancelled spinner", async () => {
    let loader: CancellableLoader
    let pid = 0
    const started = Effect.runSync(Deferred.make<void>())
    const ctx = { ui: { custom: (factory: Function) => new Promise<void>(resolve => {
      loader = factory({ requestRender: () => {} }, { fg: (_: string, text: string) => text }, {}, () => {
        expect(() => process.kill(pid, 0)).toThrow()
        resolve()
      })
    }) } } as unknown as ExtensionContext
    await Effect.runPromise(Effect.gen(function* () {
      const fiber = yield* withPiPreparation(ctx, () => Effect.scoped(Effect.gen(function* () {
        const child = yield* Command.make(process.execPath, "-e", "setInterval(() => {}, 1000)").pipe(Command.start)
        pid = child.pid
        yield* Deferred.succeed(started, undefined)
        return yield* child.exitCode
      }))).pipe(Effect.fork)
      yield* Deferred.await(started)
      loader.handleInput("\x1b")
      expect((yield* Fiber.await(fiber))._tag).toBe("Failure")
    }).pipe(Effect.provide(NodeContext.layer)))
  })
  it.each(["success", "failure", "cancel", "dispose"])("keeps Pi active and closes its native spinner on %s", async outcome => {
    let loader: CancellableLoader
    const done = vi.fn()
    const stop = vi.fn()
    const started = Effect.runSync(Deferred.make<void>())
    const ctx = { ui: { custom: (factory: Function) => new Promise<void>(resolve => {
      loader = factory({ stop, requestRender: () => {} }, { fg: (_: string, text: string) => text }, {}, () => { done(); resolve() })
    }) } } as unknown as ExtensionContext
    await Effect.runPromise(Effect.gen(function* () {
      const fiber = yield* withPiPreparation(ctx, setMessage => Effect.gen(function* () {
        expect(loader.render(80).join("")).toContain("Installing Magnitude…")
        yield* setMessage("Installing Magnitude…")
        expect(loader.render(80).join("")).toContain("Installing Magnitude…")
        yield* Deferred.succeed(started, undefined)
        if (outcome === "failure") return yield* Effect.fail("install failed")
        if (outcome !== "success") return yield* Effect.never
        expect(loader.render(80).join("")).toContain("Installing Magnitude…")
      })).pipe(Effect.fork)
      yield* Deferred.await(started)
      if (outcome === "cancel") loader.handleInput("\x1b")
      if (outcome === "dispose") yield* Fiber.interrupt(fiber)
      const exit = yield* Fiber.await(fiber)
      expect(exit._tag).toBe(outcome === "success" ? "Success" : "Failure")
    }))
    expect(stop).not.toHaveBeenCalled()
    expect(done).toHaveBeenCalledOnce()
  })
})

describe("hosted setup result", () => {
  it.each([completed, cancelled])("accepts a successful $_tag exit", async result => {
    expect(await Effect.runPromise(validateSetupTermination(result, { _tag: "Exited", code: 0 }))).toEqual(result)
  })
  it.each([
    [completed, 1], [cancelled, 1], [failed, 0],
  ] as const)("rejects inconsistent result and exit code", async (result, code) => {
    await expect(Effect.runPromise(validateSetupTermination(result, { _tag: "Exited", code }))).rejects.toThrow("consistent completion")
  })
  it("preserves a reported failure", async () => {
    await expect(Effect.runPromise(validateSetupTermination(failed, { _tag: "Exited", code: 1 }))).rejects.toThrow("installation failed")
  })
  it("does not activate a model after an unexpected signal", async () => {
    await expect(Effect.runPromise(validateSetupTermination(completed, { _tag: "Signaled", signal: "SIGTERM" }))).rejects.toThrow("consistent completion")
  })
})

describe("Pi terminal ownership", () => {
  it.each(["success", "failure", "defect"])("restores the terminal after %s", async outcome => {
    const calls: string[] = []
    const tui = {
      terminal: { write: (sequence: string) => { expect(sequence).toContain("\x1b[?1049l"); calls.push("reset") } },
      stop: () => calls.push("stop"), start: () => calls.push("start"), requestRender: () => calls.push("render"),
    }
    const ctx = { ui: { custom: (factory: Function) => new Promise<void>(resolve => {
      factory(tui, {}, {}, () => { calls.push("done"); resolve() })
    }) } } as unknown as ExtensionContext
    const work = Effect.sync(() => calls.push("work")).pipe(Effect.zipRight(
      outcome === "success" ? Effect.void : outcome === "failure" ? Effect.fail("failed") : Effect.die("defect"),
    ))
    await Effect.runPromiseExit(withPiTerminal(ctx, work))
    expect(calls).toEqual(["stop", "work", "reset", "start", "render", "done"])
  })
  it("reports a custom UI rejection without starting the child", async () => {
    const work = vi.fn()
    const ctx = { ui: { custom: () => Promise.reject(new Error("closed")) } } as unknown as ExtensionContext
    await expect(Effect.runPromise(withPiTerminal(ctx, Effect.sync(work)))).rejects.toThrow("Could not open Pi setup")
    expect(work).not.toHaveBeenCalled()
  })
  it("releases terminal ownership when its scope is interrupted", async () => {
    const calls: string[] = []
    const started = Effect.runSync(Deferred.make<void>())
    const ctx = { ui: { custom: (factory: Function) => new Promise<void>(resolve => {
      factory({ terminal: { write: () => calls.push("reset") }, stop: () => calls.push("stop"), start: () => calls.push("start"), requestRender: () => calls.push("render") }, {}, {}, () => { calls.push("done"); resolve() })
    }) } } as unknown as ExtensionContext
    await Effect.runPromise(Effect.gen(function* () {
      const fiber = yield* withPiTerminal(ctx, Deferred.succeed(started, undefined).pipe(Effect.zipRight(Effect.never))).pipe(Effect.fork)
      yield* Deferred.await(started)
      yield* Fiber.interrupt(fiber)
    }))
    expect(calls).toEqual(["stop", "reset", "start", "render", "done"])
  })
})

describe("setup command", () => {
  const harness = (run = vi.fn<PiSetup["run"]>(() => Effect.succeed(completed))) => {
    const registerCommand = vi.fn()
    const setModel = vi.fn(async () => true)
    const model = { id: modelId, provider: "magnitude" }
    const ctx = {
      mode: "tui", isIdle: () => true, hasPendingMessages: () => false,
      ui: { notify: vi.fn() },
      modelRegistry: { refresh: vi.fn(async () => {}), find: vi.fn(() => model) },
      reload: vi.fn(async () => {}),
    }
    const setup = registerMagnitudeSetup({ registerCommand, setModel } as unknown as ExtensionAPI, Layer.succeed(PiSetup, { run }))
    return { ctx, setModel, dispose: setup.dispose, run, start: () => setup.run(ctx as unknown as ExtensionContext), invoke: () => registerCommand.mock.calls[0]![1].handler("", ctx as unknown as ExtensionCommandContext) }
  }
  it("refreshes, activates the exact model, and reloads only after completion", async () => {
    const h = harness()
    try {
      await h.invoke()
      expect(h.ctx.modelRegistry.refresh).toHaveBeenCalledOnce()
      expect(h.ctx.modelRegistry.find).toHaveBeenCalledExactlyOnceWith("magnitude", modelId)
      expect(h.setModel).toHaveBeenCalledOnce()
      expect(h.ctx.reload).toHaveBeenCalledOnce()
      expect(h.ctx.ui.notify).not.toHaveBeenCalled()
    } finally { await h.dispose() }
  })
  it("activates directly from a startup event without command dispatch or reload", async () => {
    const h = harness()
    try {
      expect(await h.start()).toBe(true)
      expect(h.run).toHaveBeenCalledOnce()
      expect(h.setModel).toHaveBeenCalledOnce()
      expect(h.ctx.reload).not.toHaveBeenCalled()
    } finally { await h.dispose() }
  })
  it.each(["rpc", "busy", "queued"])("does not start for %s", async condition => {
    const h = harness()
    if (condition === "rpc") h.ctx.mode = "rpc"
    if (condition === "busy") h.ctx.isIdle = () => false
    if (condition === "queued") h.ctx.hasPendingMessages = () => true
    try {
      await h.invoke()
      expect(h.run).not.toHaveBeenCalled()
      expect(h.ctx.ui.notify).toHaveBeenCalledOnce()
      expect(h.ctx.reload).not.toHaveBeenCalled()
    } finally { await h.dispose() }
  })
  it("leaves the existing model alone on cancellation", async () => {
    const h = harness(vi.fn(() => Effect.succeed(cancelled)))
    try {
      await h.invoke()
      expect(h.setModel).not.toHaveBeenCalled()
      expect(h.ctx.modelRegistry.refresh).not.toHaveBeenCalled()
      expect(h.ctx.reload).not.toHaveBeenCalled()
    } finally { await h.dispose() }
  })
  it("reports activation failure without reloading", async () => {
    const h = harness()
    h.setModel.mockResolvedValue(false)
    try {
      await h.invoke()
      expect(h.ctx.ui.notify.mock.calls[0]![0]).toContain("could not activate")
      expect(h.ctx.reload).not.toHaveBeenCalled()
    } finally { await h.dispose() }
  })
  it("reports reload failure as a distinct post-install phase", async () => {
    const h = harness()
    h.ctx.reload.mockRejectedValue(new Error("reload failed"))
    try {
      await expect(h.invoke()).rejects.toThrow("completed and selected the model")
      expect(h.run).toHaveBeenCalledOnce()
      expect(h.setModel).toHaveBeenCalledOnce()
    } finally { await h.dispose() }
  })
  it("admits only one concurrent setup", async () => {
    const started = Effect.runSync(Deferred.make<void>())
    const finish = Effect.runSync(Deferred.make<void>())
    const h = harness(vi.fn(() => Deferred.succeed(started, undefined).pipe(
      Effect.zipRight(Deferred.await(finish)), Effect.as(cancelled),
    )))
    try {
      const first = h.invoke()
      await Effect.runPromise(Deferred.await(started))
      await h.invoke()
      expect(h.run).toHaveBeenCalledOnce()
      expect(h.ctx.ui.notify).toHaveBeenCalledExactlyOnceWith("Magnitude setup is already open.", "error")
      await Effect.runPromise(Deferred.succeed(finish, undefined))
      await first
    } finally { await h.dispose() }
  })
  it("reports setup failure without changing the registry", async () => {
    const h = harness(vi.fn(() => Effect.fail(new PiSetupFailed({ message: "failed" }))))
    try {
      await h.invoke()
      expect(h.ctx.ui.notify).toHaveBeenCalledExactlyOnceWith("failed", "error")
      expect(h.ctx.modelRegistry.refresh).not.toHaveBeenCalled()
    } finally { await h.dispose() }
  })
})
