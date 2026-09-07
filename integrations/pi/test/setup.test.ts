import { Deferred, Effect, Fiber, Layer, Schema } from "effect"
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent"
import { describe, expect, it, vi } from "vitest"
import { ConnectionClosed, MagnitudeClient, ProviderModelIdSchema } from "@magnitudedev/sdk"
import { FileSystem } from "@effect/platform"
import * as Command from "@effect/platform/Command"
import { NodeContext } from "@effect/platform-node"
import { PiSetup, PiSetupFailed, PiSetupLive, prepareMagnitudeCli, readSetupModel, registerMagnitudeSetup, validateSetupTermination, withPiTerminal, withPiPreparation } from "../extensions/setup"
import type { CancellableLoader } from "@earendil-works/pi-tui"

const modelId = ProviderModelIdSchema.make("test:gguf:q4")
const completed = { _tag: "Completed", modelId } as const
const cancelled = { _tag: "Cancelled" } as const
const encodeString = Schema.encodeSync(Schema.parseJson(Schema.String))

describe("actual setup child boundary", () => {
  it.each(["cancelled", "failed", "signal", "old-cli"])("restores Pi after %s without model lookup", async scenario => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "pi-setup-fixture-" })
      const executable = `${directory}/magnitude`
      yield* fs.writeFileString(executable, `#!/usr/bin/env node
if (process.argv[2] === '--version') process.exit(0)
if (process.argv.slice(2).join(' ') !== 'setup --host pi') process.exit(2)
if (${encodeString(scenario)} === 'signal') process.kill(process.pid, 'SIGKILL')
process.exit(${scenario === "cancelled" ? 130 : 1})
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
      const result = yield* Effect.flatMap(PiSetup, setup => setup.run(ctx)).pipe(Effect.provide(PiSetupLive), Effect.either)
      if (scenario === "cancelled") expect(result).toMatchObject({ _tag: "Right", right: cancelled })
      else expect(result._tag).toBe("Left")
      expect(calls).toEqual(["stop", "reset", "start"])
      expect(yield* fs.readDirectory(directory)).toEqual(["magnitude"])
    })).pipe(Effect.provide(NodeContext.layer)))
  })
})

describe("first-encounter CLI installation", () => {
  it.each(["missing", "existing", "old-version", "nonzero", "override", "permission", "npm-failed", "npm-missing", "interrupted", "bad-install"])("handles %s without an ambient installation", async scenario => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "pi-cli-bootstrap-" })
      const executable = `${directory}/magnitude`
      const receipt = `${directory}/install.json`
      const cli = `#!${process.execPath}\nif (process.argv[2] !== '--version') process.exit(2)\nconsole.log('0.0.1')\n`
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
      if (["existing", "old-version", "nonzero", "permission"].includes(scenario)) {
        yield* fs.writeFileString(executable, scenario === "old-version" ? `#!${process.execPath}\nconsole.log('0.0.0')\n` : cli + (scenario === "nonzero" ? "process.exitCode = 1\n" : ""))
        yield* fs.chmod(executable, scenario === "permission" ? 0o644 : 0o755)
      }
      if (scenario !== "npm-missing") {
        yield* fs.writeFileString(`${directory}/npm`, `#!${process.execPath}
const fs = require('node:fs')
fs.writeFileSync(${encodeString(receipt)}, JSON.stringify(process.argv.slice(2)))
console.log('hidden npm output')
if (${encodeString(scenario)} === 'npm-failed') { console.error('EACCES: test prefix not writable'); process.exit(1) }
if (${encodeString(scenario)} === 'interrupted') process.kill(process.pid, 'SIGTERM')
fs.writeFileSync(${encodeString(executable)}, ${encodeString(scenario === "bad-install" ? `#!${process.execPath}\nprocess.exit(1)\n` : cli)}, { mode: 0o755 })
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
      if (scenario === "missing" || scenario === "existing" || scenario === "old-version") {
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

describe("setup exit status", () => {
  it.each([[0, true], [130, false]] as const)("interprets exit %i", async (code, expected) => {
    expect(await Effect.runPromise(validateSetupTermination({ _tag: "Exited", code }))).toBe(expected)
  })
  it.each([1, 2, 127])("rejects failure exit %i", async code => {
    await expect(Effect.runPromise(validateSetupTermination({ _tag: "Exited", code }))).rejects.toThrow("setup failed")
  })
  it("treats SIGINT as cancellation", async () => {
    expect(await Effect.runPromise(validateSetupTermination({ _tag: "Signaled", signal: "SIGINT" }))).toBe(false)
  })
  it("reports an unexpected signal", async () => {
    await expect(Effect.runPromise(validateSetupTermination({ _tag: "Signaled", signal: "SIGTERM" }))).rejects.toThrow("unexpectedly")
  })
})

describe("setup SDK lookup", () => {
  it("reports a failed lookup without retrying setup", async () => {
    const getSlots = vi.fn(() => Effect.fail(new ConnectionClosed({})))
    await expect(Effect.runPromise(readSetupModel.pipe(
      Effect.provideService(MagnitudeClient, { models: { getSlots } } as unknown as MagnitudeClient),
    ))).rejects.toThrow("Magnitude setup completed, but the selected model could not be read: Magnitude client is closed. Run /reload and select it with /model.")
    expect(getSlots).toHaveBeenCalledExactlyOnceWith({})
  })
  it.each(["ConfiguredLocal", "Unassigned", "Resolving", "ConfiguredRemote"])("handles %s primary model", async tag => {
    const getSlots = vi.fn(() => Effect.succeed({ slots: { primary: { _tag: tag, selection: { providerModelId: modelId } } } }))
    const result = await Effect.runPromise(readSetupModel.pipe(
      Effect.provideService(MagnitudeClient, { models: { getSlots } } as unknown as MagnitudeClient),
      Effect.either,
    ))
    expect(getSlots).toHaveBeenCalledExactlyOnceWith({})
    if (tag === "ConfiguredLocal") expect(result).toMatchObject({ _tag: "Right", right: modelId })
    else expect(result).toMatchObject({ _tag: "Left", left: { message: expect.stringContaining("no local primary model") } })
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
