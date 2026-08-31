import { Command } from "@commander-js/extra-typings"
import { describe, expect, it } from "vitest"
import { registerConnectionsCommand } from "./connections"

describe("connections command contract", () => {
  it("connects all models and optionally changes the harness current selection", () => {
    const program = new Command()
    registerConnectionsCommand(program)
    const connections = program.commands.find((command) => command.name() === "connections")
    const add = connections?.commands.find((command) => command.name() === "add")
    expect(add?.registeredArguments.map((argument) => argument.name())).toEqual(["harness"])
    expect(add?.options.map(({ long }) => long)).toContain("--set-current")
    expect(add?.options.map(({ long }) => long)).toContain("--install-skill")
    expect(add?.description()).toBe("Connect every installed Magnitude model to a harness")
  })
})
