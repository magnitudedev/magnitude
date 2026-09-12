import { Command } from "@commander-js/extra-typings"
import { HarnessIdSchema } from "@magnitudedev/client-common"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { describe, expect, it, vi } from "vitest"
import { Option } from "effect"
import { registerConnectionsCommand } from "./connections"
import { addConnection, syncConnections, renderAddedConnection, renderConnections } from "./connections-runtime"

const startupProbe = vi.hoisted(() => vi.fn(() => { throw new Error("Unexpected service startup") }))
vi.mock("../server/acn-connection", async () => {
  const { Effect } = await import("effect")
  return { headlessAcnConnection: Effect.sync(startupProbe) }
})

describe("connections command contract", () => {
  it.each([
    [() => addConnection("magnitude", undefined, false), "Unsupported harness: magnitude"],
    [() => syncConnections("magnitude"), "Unsupported harness: magnitude"],
    [() => addConnection("pi", "", false), "Invalid model ID: "],
  ])("rejects invalid arguments before service startup", async (command, message) => {
    const exitCode = process.exitCode
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true)
    startupProbe.mockClear()
    try {
      await command()
      expect(stderr).toHaveBeenCalledWith(`${message}\n`)
      expect(process.exitCode).toBe(1)
      expect(startupProbe).not.toHaveBeenCalled()
    } finally {
      stderr.mockRestore()
      process.exitCode = exitCode
    }
  })

  it("connects all models and optionally selects a harness model", () => {
    const program = new Command()
    registerConnectionsCommand(program)
    const connections = program.commands.find((command) => command.name() === "connections")
    const add = connections?.commands.find((command) => command.name() === "add")
    expect(add?.registeredArguments.map((argument) => argument.name())).toEqual(["harness"])
    expect(add?.options.map(({ long }) => long)).toContain("--set-model")
    expect(add?.options.map(({ long }) => long)).not.toContain("--set-current")
    expect(add?.options.map(({ long }) => long)).toContain("--install-skill")
    expect(add?.description()).toBe("Connect installed Magnitude models to a harness")
  })

  it("distinguishes configuration integrity from installation", () => {
    const common = { configurationFiles: [], plugin: Option.none(), managed: false }
    const output = renderConnections([
      { ...common, id: HarnessIdSchema.make("codex"), name: "Codex", installed: true, inspection: { _tag: "Connected" } },
      { ...common, id: HarnessIdSchema.make("claude-code"), name: "Claude Code", installed: true, inspection: { _tag: "Disconnected", reason: "Settings changed" } },
      { ...common, id: HarnessIdSchema.make("cline"), name: "Cline", installed: false, inspection: { _tag: "Disconnected", reason: "No configuration" } },
      { ...common, id: HarnessIdSchema.make("pi"), name: "Pi", installed: true, inspection: { _tag: "Unavailable", reason: "Permission denied" } },
    ])
    expect(output).toContain("Connected")
    expect(output).toContain("Disconnected")
    expect(output).toContain("Not installed")
    expect(output).toContain("Unable to check")
    expect(output).toContain("Settings changed")
    expect(output).not.toContain("Built in")
  })

  it.each([
    [{ _tag: "Connected" } as const, "Connected"],
    [{ _tag: "Disconnected", reason: "Settings changed" } as const, "Disconnected"],
    [{ _tag: "Unavailable", reason: "Permission denied" } as const, "Unable to check"],
  ])("preserves configuration status when the executable is missing: %s", (inspection, status) => {
    const output = renderConnections([{
      id: HarnessIdSchema.make("claude-code"), name: "Claude Code", installed: false,
      configurationFiles: [], plugin: Option.none(), managed: false, inspection,
    }])
    expect(output).toContain("Not installed")
    expect(output).toContain(status)
    if (inspection._tag !== "Connected") expect(output).toContain(inspection.reason)
  })

  it("reports automatic Pi package and skill installation in the headless flow", () => {
    const pi = HarnessIdSchema.make("pi")
    const model = ProviderModelIdSchema.make("local/model")
    const output = renderAddedConnection({
      harness: pi,
      model: Option.some(model),
      connection: {
        companion: Option.some({
          name: "Magnitude for Pi",
          source: "npm:@magnitudedev/pi-extension@0.0.1",
          securityNotice: "Pi extensions execute with your user permissions.",
          status: "installed",
          activationInstructions: Option.some("Restart existing Pi sessions or run /reload to activate the extension."),
        }),
        skillInstalled: true,
        startupInstalled: false,
      },
    })

    expect(output).toContain("Connected pi to Magnitude.")
    expect(output).toContain("Magnitude for Pi  Installed")
    expect(output).toContain("Skill             Installed")
    expect(output).not.toContain("pi --model")
    expect(output).toContain("run /reload to activate the extension")
  })
})
