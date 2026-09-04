import { Command } from "@commander-js/extra-typings"
import { HarnessIdSchema } from "@magnitudedev/client-common"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { registerConnectionsCommand } from "./connections"
import { renderAddedConnection, renderConnections, renderLaunchPlan } from "./connections-runtime"

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
      { id: HarnessIdSchema.make("gptme"), name: "gptme", availability: "Installed", selectable: true, connected: false },
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
      launchPlan: Option.some({
        harness: pi,
        command: "pi",
        executable: "/installed/pi",
        args: ["--model", `magnitude/${model}`],
        environment: {},
        modelId: model,
      }),
    })

    expect(output).toContain("Connected pi to Magnitude.")
    expect(output).toContain("Magnitude for Pi  Installed")
    expect(output).toContain("Skill             Installed")
    expect(output).toContain("pi --model magnitude/local/model")
    expect(output).toContain("run /reload to activate the extension")
  })
})
