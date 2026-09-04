import type {
  ExtensionAPI,
  ExtensionCommandContext,
  RegisteredCommand,
} from "@earendil-works/pi-coding-agent"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { registerMagnitudeCommands } from "../extensions/commands"

type Command = Omit<RegisteredCommand, "name" | "sourceInfo">

const success = (command: string, data: unknown) => JSON.stringify({
  schemaVersion: 1,
  command,
  ok: true,
  data,
})

const model = {
  modelId: "local/model:q4",
  displayName: "Local Model (Q4)",
  installation: "installed",
  residency: "unloaded",
}

const setup = () => {
  const commands = new Map<string, Command>()
  const exec = vi.fn()
  const pi = {
    exec,
    registerCommand: (name: string, command: Command) => commands.set(name, command),
  } as unknown as ExtensionAPI
  registerMagnitudeCommands(pi)
  const ui = { notify: vi.fn(), select: vi.fn() }
  const ctx = { ui } as unknown as ExtensionCommandContext
  return { commands, exec, ui, ctx }
}

beforeEach(() => {
  delete process.env.MAGNITUDE_CLI
})

describe("Magnitude slash commands", () => {
  it("registers all commands and renders model state from the versioned JSON contract", async () => {
    const { commands, exec } = setup()
    expect([...commands.keys()]).toEqual(["load-model", "stop-model"])
    exec.mockResolvedValue({
      code: 0,
      stdout: success("models.status", { state: "ready", models: [model] }),
      stderr: "",
      killed: false,
    })
    const load = commands.get("load-model")!
    expect(await load.getArgumentCompletions!("")).toEqual([{
      value: model.modelId,
      label: model.displayName,
      description: "unloaded",
    }])

    expect(exec).toHaveBeenCalledWith("magnitude", ["models", "status", "--json"], { timeout: 120_000 })
  })

  it("loads an explicit model, stops the resident model, and honors MAGNITUDE_CLI", async () => {
    process.env.MAGNITUDE_CLI = "/opt/magnitude-dev"
    const { commands, exec, ui, ctx } = setup()
    exec
      .mockResolvedValueOnce({
        code: 0,
        stdout: success("models.load", { modelId: model.modelId }),
        stderr: "",
        killed: false,
      })
      .mockResolvedValueOnce({
        code: 0,
        stdout: success("models.stop", {}),
        stderr: "",
        killed: false,
      })

    await commands.get("load-model")!.handler(model.modelId, ctx)
    await commands.get("stop-model")!.handler("", ctx)

    expect(exec).toHaveBeenNthCalledWith(1, "/opt/magnitude-dev", ["models", "load", model.modelId, "--json"], { timeout: 120_000 })
    expect(exec).toHaveBeenNthCalledWith(2, "/opt/magnitude-dev", ["models", "stop", "--json"], { timeout: 120_000 })
    expect(ui.notify).toHaveBeenNthCalledWith(1, `Loaded ${model.modelId}.`, "info")
    expect(ui.notify).toHaveBeenNthCalledWith(2, "Stopped the active Magnitude model.", "info")
  })

  it("selects a load target and exposes cached argument completions", async () => {
    const { commands, exec, ui, ctx } = setup()
    exec
      .mockResolvedValueOnce({
        code: 0,
        stdout: success("models.status", { state: "ready", models: [model] }),
        stderr: "",
        killed: false,
      })
      .mockResolvedValueOnce({
        code: 0,
        stdout: success("models.load", { modelId: model.modelId }),
        stderr: "",
        killed: false,
      })
    ui.select.mockResolvedValue("Local Model (Q4) · unloaded · local/model:q4")

    const load = commands.get("load-model")!
    expect(await load.getArgumentCompletions?.("model")).toEqual([{
      value: model.modelId,
      label: model.displayName,
      description: "unloaded",
    }])
    await load.handler("", ctx)
    expect(exec).toHaveBeenLastCalledWith("magnitude", ["models", "load", model.modelId, "--json"], { timeout: 120_000 })
  })

  it("surfaces structured CLI failures and rejects incompatible success JSON", async () => {
    const { commands, exec, ui, ctx } = setup()
    exec.mockResolvedValueOnce({
      code: 1,
      stdout: "",
      stderr: JSON.stringify({ schemaVersion: 1, command: "models.stop", ok: false, error: { message: "No model is running" } }),
      killed: false,
    })
    await commands.get("stop-model")!.handler("", ctx)
    expect(ui.notify).toHaveBeenLastCalledWith("No model is running", "error")

    exec.mockResolvedValueOnce({ code: 0, stdout: "{}", stderr: "", killed: false })
    await commands.get("stop-model")!.handler("", ctx)
    expect(ui.notify).toHaveBeenLastCalledWith("This Magnitude extension requires a newer Magnitude CLI.", "error")

    exec.mockResolvedValueOnce({
      code: 1,
      stdout: "",
      stderr: "error: unknown option '--json'\n",
      killed: false,
    })
    await commands.get("stop-model")!.handler("", ctx)
    expect(ui.notify).toHaveBeenLastCalledWith("This Magnitude extension requires a newer Magnitude CLI.", "error")

    exec.mockResolvedValueOnce({
      code: 0,
      stdout: JSON.stringify({ schemaVersion: 2, command: "models.stop", ok: true, data: {} }),
      stderr: "",
      killed: false,
    })
    await commands.get("stop-model")!.handler("", ctx)
    expect(ui.notify).toHaveBeenLastCalledWith("This Magnitude extension requires a newer Magnitude CLI.", "error")

    exec.mockResolvedValueOnce({ code: 0, stdout: "not json", stderr: "", killed: false })
    await commands.get("stop-model")!.handler("", ctx)
    expect(ui.notify).toHaveBeenLastCalledWith("Magnitude returned invalid JSON", "error")

    exec.mockRejectedValueOnce(new Error("Executable not found: magnitude"))
    await commands.get("stop-model")!.handler("", ctx)
    expect(ui.notify).toHaveBeenLastCalledWith("Executable not found: magnitude", "error")

    exec.mockResolvedValueOnce({
      code: 1,
      stdout: "",
      stderr: JSON.stringify({ schemaVersion: 1, command: "models.load", ok: false, error: { message: "wrong command" } }),
      killed: false,
    })
    await commands.get("stop-model")!.handler("", ctx)
    expect(ui.notify).toHaveBeenLastCalledWith("Magnitude returned an incompatible JSON error response", "error")
  })

  it("does not offer models that have no loadable installation", async () => {
    const { commands, exec, ui, ctx } = setup()
    exec.mockResolvedValue({
      code: 0,
      stdout: success("models.status", {
        state: "ready",
        models: [{
          ...model,
          installation: "not_installed",
          residency: undefined,
        }],
      }),
      stderr: "",
      killed: false,
    })

    const load = commands.get("load-model")!
    expect(await load.getArgumentCompletions?.("")).toEqual([])
    await load.handler("", ctx)

    expect(ui.select).not.toHaveBeenCalled()
    expect(ui.notify).toHaveBeenLastCalledWith("No installed Magnitude models are available.", "info")
    expect(exec).toHaveBeenCalledTimes(1)
  })
})
