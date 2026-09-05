import type {
  ExtensionAPI,
  ExtensionCommandContext,
  RegisteredCommand,
} from "@earendil-works/pi-coding-agent"
import { afterEach, describe, expect, it, vi } from "vitest"
import { Deferred, Effect, Option } from "effect"
import { ProtocolMismatch, ServiceUnavailable, type ModelCatalogState } from "@magnitudedev/sdk"
import { registerMagnitudeCommands } from "../extensions/commands"
import { fakeAcnImplementationsLayer } from "../../../packages/client-common/src/state/fake-acn-implementations"
import { makeInstalledCatalogModel, makeCatalogOnlyModel } from "../../../cli/src/features/local-inference/test-fixtures"

type Command = Omit<RegisteredCommand, "name" | "sourceInfo">
const installed = makeInstalledCatalogModel()
const ready = (models = [installed]): ModelCatalogState => ({
  _tag: "Ready", providers: [], failures: [],
  models: models.map(product => ({ _tag: "Local", product, offering: Option.none() })),
  localModelPreparation: { discovery: { complete: true, modelsFound: models.length }, assessment: { complete: true, settledModels: models.length, totalModels: models.length } },
})
const disposers: (() => Promise<void>)[] = []
afterEach(async () => {
  await Promise.all(disposers.splice(0).map(dispose => dispose()))
  vi.unstubAllEnvs()
})
const setup = (execute = vi.fn<(name: string, input: unknown) => Effect.Effect<unknown, unknown>>(() => Effect.succeed(ready()))) => {
  const commands = new Map<string, Command>()
  const exec = vi.fn<ExtensionAPI["exec"]>(async () => ({ code: 0, stdout: "", stderr: "", killed: false }))
  const pi = {
    exec,
    registerCommand: (name: string, command: Command) => commands.set(name, command),
  } as unknown as ExtensionAPI
  const dispose = registerMagnitudeCommands(pi, fakeAcnImplementationsLayer(execute))
  disposers.push(dispose)
  const ui = { notify: vi.fn(), select: vi.fn() }
  const reload = vi.fn(async () => {})
  return { commands, execute, exec, ui, reload, ctx: { ui, reload } as unknown as ExtensionCommandContext, dispose }
}

describe("Magnitude SDK slash commands", () => {
  it("is passive until used and loads/stops through RPC without shell calls", async () => {
    const { commands, execute, exec, ui, ctx } = setup()
    expect(execute).not.toHaveBeenCalled()
    execute.mockImplementation(() => Effect.succeed({}))
    await commands.get("load-model")!.handler(installed.modelId, ctx)
    await commands.get("stop-model")!.handler("", ctx)
    expect(execute.mock.calls).toEqual([["LoadLocalModel", { modelId: installed.modelId }], ["StopActiveLocalModel", {}]])
    expect(exec).not.toHaveBeenCalled()
    expect(ui.notify).toHaveBeenNthCalledWith(1, `Loaded ${installed.modelId}.`, "info")
    expect(ui.notify).toHaveBeenNthCalledWith(2, "Stopped the active Magnitude model.", "info")
  })

  it("does not cache initialization and shares in-flight completions", async () => {
    const { commands, execute, ui, ctx } = setup()
    const load = commands.get("load-model")!
    execute.mockReturnValueOnce(Effect.succeed({ _tag: "Initializing" }))
    await load.handler("", ctx)
    expect(ui.notify).toHaveBeenCalledWith("Magnitude is discovering local models. Try again shortly.", "info")
    const pending = Effect.runSync(Deferred.make<ModelCatalogState>())
    execute.mockReturnValueOnce(Deferred.await(pending))
    const first = load.getArgumentCompletions!("")
    const second = load.getArgumentCompletions!(installed.modelId)
    await vi.waitFor(() => expect(execute).toHaveBeenCalledTimes(2))
    Effect.runSync(Deferred.succeed(pending, ready()))
    expect(await first).toHaveLength(1)
    expect(await second).toHaveLength(1)
    await load.getArgumentCompletions!("")
    expect(execute).toHaveBeenCalledTimes(2)
  })

  it("refreshes selection, filters uninstalled models, and invalidates after mutation", async () => {
    const { commands, execute, ui, ctx } = setup()
    const load = commands.get("load-model")!
    execute.mockReturnValueOnce(Effect.succeed(ready([makeCatalogOnlyModel()]))).mockReturnValue(Effect.succeed(ready()))
    expect(await load.getArgumentCompletions!("")).toEqual([])
    ui.select.mockImplementation((_title: string, labels: string[]) => Promise.resolve(labels[0]))
    await load.handler("", ctx)
    await load.getArgumentCompletions!("")
    expect(execute.mock.calls.map(([name]) => name)).toEqual(["GetModelCatalog", "GetModelCatalog", "LoadLocalModel", "GetModelCatalog"])
  })

  it("syncs through the selected CLI and reloads on mismatch without replaying the mutation", async () => {
    vi.stubEnv("MAGNITUDE_CLI", "/custom magnitude/bin/magnitude")
    const { commands, execute, exec, ui, reload, ctx, dispose } = setup()
    // The real host disposes the old extension as part of reload.
    reload.mockImplementation(dispose)
    execute.mockReturnValue(Effect.fail(new ProtocolMismatch({ expected: 2, actual: 3, daemonVersion: "1.2.3" })))
    await commands.get("load-model")!.handler(installed.modelId, ctx)
    expect(exec).toHaveBeenCalledExactlyOnceWith("/custom magnitude/bin/magnitude", ["connections", "sync", "pi"], { signal: expect.any(AbortSignal), timeout: 120_000 })
    expect(reload).toHaveBeenCalledTimes(1)
    expect(ui.notify).toHaveBeenLastCalledWith(expect.stringContaining("retry your model command"), "info")
    expect(execute).toHaveBeenCalledTimes(1)
  })

  it("does not sync from autocomplete, but repairs on the next explicit command", async () => {
    const { commands, execute, exec, reload, ui, ctx } = setup()
    execute.mockReturnValue(Effect.fail(new ProtocolMismatch({ expected: 2, actual: 3, daemonVersion: "1.2.3" })))
    expect(await commands.get("load-model")!.getArgumentCompletions!("")).toBeNull()
    expect(exec).not.toHaveBeenCalled()
    expect(reload).not.toHaveBeenCalled()
    expect(ui.notify).not.toHaveBeenCalled()
    await commands.get("stop-model")!.handler("", ctx)
    expect(exec).toHaveBeenCalledTimes(1)
    expect(reload).toHaveBeenCalledTimes(1)
  })

  it("runs only one sync and reload for concurrent mismatches", async () => {
    const { commands, execute, exec, reload, ctx } = setup()
    execute.mockReturnValue(Effect.fail(new ProtocolMismatch({ expected: 2, actual: 3, daemonVersion: "1.2.3" })))
    const synced = Effect.runSync(Deferred.make<Awaited<ReturnType<ExtensionAPI["exec"]>>>())
    exec.mockImplementation(() => Effect.runPromise(Deferred.await(synced)))
    const first = commands.get("load-model")!.handler(installed.modelId, ctx)
    const second = commands.get("stop-model")!.handler("", ctx)
    await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(1))
    expect(reload).not.toHaveBeenCalled()
    Effect.runSync(Deferred.succeed(synced, { code: 0, stdout: "", stderr: "", killed: false }))
    await Promise.all([first, second])
    expect(exec).toHaveBeenCalledTimes(1)
    expect(reload).toHaveBeenCalledTimes(1)
    expect(execute).toHaveBeenCalledTimes(2)
  })

  it.each([
    { code: 1, stdout: "", stderr: "CLI RPC does not match the daemon", killed: false },
    { code: 0, stdout: "", stderr: "", killed: true },
  ])("reports unsuccessful sync without reloading or looping ($code, killed=$killed)", async result => {
    const { commands, execute, exec, reload, ui, ctx } = setup()
    execute.mockReturnValue(Effect.fail(new ProtocolMismatch({ expected: 2, actual: 3, daemonVersion: "1.2.3" })))
    exec.mockResolvedValue(result)
    await commands.get("load-model")!.handler(installed.modelId, ctx)
    expect(ui.notify).toHaveBeenLastCalledWith(expect.stringContaining("Magnitude plugin sync failed"), "error")
    await commands.get("stop-model")!.handler("", ctx)
    expect(ui.notify).toHaveBeenLastCalledWith(expect.stringContaining("already attempted"), "error")
    expect(exec).toHaveBeenCalledTimes(1)
    expect(reload).not.toHaveBeenCalled()
  })

  it("reports a missing CLI without reloading", async () => {
    const { commands, execute, exec, reload, ui, ctx } = setup()
    execute.mockReturnValue(Effect.fail(new ProtocolMismatch({ expected: 2, actual: 3, daemonVersion: "1.2.3" })))
    exec.mockRejectedValue(new Error("spawn magnitude ENOENT"))
    await commands.get("stop-model")!.handler("", ctx)
    expect(ui.notify).toHaveBeenLastCalledWith(expect.stringContaining("ENOENT"), "error")
    expect(reload).not.toHaveBeenCalled()
  })

  it("does not repair non-mismatch failures", async () => {
    const { commands, execute, exec, reload, ui, ctx } = setup()
    execute.mockReturnValue(Effect.fail(new ServiceUnavailable({ origin: "http://localhost:10100", message: "unavailable" })))
    await commands.get("stop-model")!.handler("", ctx)
    expect(ui.notify).toHaveBeenLastCalledWith("unavailable", "error")
    expect(exec).not.toHaveBeenCalled()
    expect(reload).not.toHaveBeenCalled()
  })

  it("disposal aborts sync without reloading or posting a stale result", async () => {
    const { commands, execute, exec, reload, ui, ctx, dispose } = setup()
    execute.mockReturnValue(Effect.fail(new ProtocolMismatch({ expected: 2, actual: 3, daemonVersion: "1.2.3" })))
    exec.mockImplementation((_command, _args, options) => new Promise((_resolve, reject) => {
      options?.signal?.addEventListener("abort", () => reject(new Error("cancelled")), { once: true })
    }))
    const pending = commands.get("stop-model")!.handler("", ctx).catch(() => undefined)
    await vi.waitFor(() => expect(exec).toHaveBeenCalledTimes(1))
    await dispose()
    await pending
    expect(exec.mock.calls[0]?.[2]?.signal?.aborted).toBe(true)
    expect(reload).not.toHaveBeenCalled()
    expect(ui.notify).toHaveBeenCalledTimes(1)
  })

  it("disposal interrupts RPC work without a stale notification", async () => {
    let interrupted = false
    const { commands, execute, ui, ctx, dispose } = setup()
    execute.mockReturnValue(Effect.never.pipe(Effect.ensuring(Effect.sync(() => { interrupted = true }))))
    const pending = commands.get("load-model")!.handler(installed.modelId, ctx).catch(() => undefined)
    await vi.waitFor(() => expect(execute).toHaveBeenCalledTimes(1))
    await dispose()
    await pending
    expect(interrupted).toBe(true)
    expect(ui.notify).not.toHaveBeenCalled()
  })
})
