import { describe, expect, it } from "vitest"
import { Command } from "@commander-js/extra-typings"
import { Option } from "effect"
import { registerServiceCommand } from "./server"
import { renderServiceStatus } from "./server-runtime"

describe("Magnitude service definitions", () => {
  it("registers only the public service command group", () => {
    const program = new Command().name("magnitude")
    registerServiceCommand(program)
    expect(program.commands.map((command) => command.name())).toEqual(["native-runtime-check", "service"])
    expect(program.commands[1]!.commands.map((command) => command.name())).toEqual([
      "install",
      "uninstall",
      "start",
      "stop",
      "status",
    ])
  })
  it("renders service status as a labeled product summary", () => {
    expect(renderServiceStatus({
      status: "Ready",
      address: "127.0.0.1:10100",
      version: Option.some("0.0.2"),
      startsAutomaticallyOnLogin: Option.some(true),
      activeModel: { _tag: "Observed", model: Option.none() },
      tray: Option.some({ _tag: "Registered" }),
    })).toBe([
      "Magnitude service",
      "  Runtime         Ready",
      "  Tray            Registered",
      "  Starts at login Yes",
      "  Version         0.0.2",
      "  Address         127.0.0.1:10100",
      "  Active model    None",
      "",
    ].join("\n"))
  })
  it("does not present an unavailable model observation as an empty runtime", () => {
    const output = renderServiceStatus({
      status: "Ready", address: "127.0.0.1:11101", version: Option.none(),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Unavailable" },
      tray: Option.some({ _tag: "Unavailable", message: "Desktop panel is unavailable" }),
    })
    expect(output).toContain("Active model    Unavailable")
    expect(output).not.toContain("Active model    None")
    expect(output).toContain("Tray            Unavailable · Desktop panel is unavailable")
  })

})
