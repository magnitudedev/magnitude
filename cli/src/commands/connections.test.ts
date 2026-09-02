import { Command } from "@commander-js/extra-typings"
import { HarnessIdSchema } from "@magnitudedev/client-common"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { describe, expect, it } from "vitest"
import { registerConnectionsCommand } from "./connections"
import { renderConnections, renderLaunchPlan } from "./connections-runtime"

describe("connections command contract", () => {
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

  it("distinguishes durable connections from installed harnesses", () => {
    const output = renderConnections([
      { id: HarnessIdSchema.make("magnitude"), name: "Magnitude Harness", availability: "Installed", selectable: true, connected: false },
      { id: HarnessIdSchema.make("codex"), name: "Codex", availability: "Installed", selectable: true, connected: true },
      { id: HarnessIdSchema.make("claude-code"), name: "Claude Code", availability: "Installed", selectable: true, connected: false },
      { id: HarnessIdSchema.make("cline"), name: "Cline", availability: "Not installed", selectable: false, connected: false },
    ])
    expect(output).toContain("Built in")
    expect(output).toContain("Connected")
    expect(output).toContain("Available")
    expect(output).toContain("Not installed")
  })

  it("renders handoff with the ambient command instead of the detected executable path", () => {
    const detectedExecutable = "/Applications/Codex.app/Contents/MacOS/codex"
    const output = renderLaunchPlan({
      harness: HarnessIdSchema.make("codex"),
      command: "codex",
      executable: detectedExecutable,
      args: ["--model", "magnitude-local/example"],
      environment: {},
      modelId: ProviderModelIdSchema.make("example"),
    })

    expect(output).toContain("codex")
    expect(output).not.toContain(detectedExecutable)
  })
})
