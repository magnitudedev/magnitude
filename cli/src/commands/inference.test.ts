import { Command } from "@commander-js/extra-typings"
import { describe, expect, it } from "vitest"
import { registerInferenceCommands } from "./inference"

describe("inference command surface", () => {
  it("exposes acquisition under catalog and residency under models", () => {
    const program = new Command().name("magnitude")
    registerInferenceCommands(program)

    expect(program.commands.map((command) => command.name())).toEqual([
      "catalog",
      "models",
    ])
    expect(program.commands[0]!.commands.map((command) => command.name())).toEqual([
      "list",
      "pull",
      "remove",
      "cancel",
    ])
    expect(program.commands[1]!.commands.map((command) => command.name())).toEqual([
      "status",
      "load",
      "stop",
    ])
    expect(program.commands[1]!.commands[2]!.registeredArguments).toHaveLength(0)
  })
})
