import { Command } from "@commander-js/extra-typings"
import { describe, expect, it } from "vitest"
import { registerInteractiveCommand } from "./interactive"

describe("interactive CLI options", () => {
  it("exposes only supported session inputs", () => {
    const program = new Command().name("magnitude")
    registerInteractiveCommand(program)

    const optionFlags = program.options.map((option) => option.flags)

    expect(optionFlags).toEqual([
      "--resume [id]",
      "--prompt <text>",
      "--atif <path>",
      "--system-override <text>",
    ])
  })

  it("keeps setup as an interactive top-level command", () => {
    const program = new Command().name("magnitude")
    registerInteractiveCommand(program)

    const setup = program.commands.find((command) => command.name() === "setup")

    expect(setup).toBeDefined()
    expect(setup!.registeredArguments).toHaveLength(0)
  })
})
