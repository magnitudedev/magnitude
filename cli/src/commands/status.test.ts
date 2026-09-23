import { describe, expect, it } from "vitest"
import { Command } from "@commander-js/extra-typings"
import { Option } from "effect"
import { registerStatusCommand } from "./status"
import { renderStatus } from "./status-runtime"

describe("Magnitude service definitions", () => {
  it("registers passive status", () => {
    const program = new Command().name("magnitude")
    registerStatusCommand(program)
    expect(program.commands.map((command) => command.name())).toEqual(["status"])
  })
  it("renders service status as a labeled product summary", () => {
    expect(renderStatus({
      status: "Ready",
      address: "127.0.0.1:10100",
      version: Option.some("0.0.2"),
      startsAutomaticallyOnLogin: Option.some(true),
      activeModel: { _tag: "Observed", model: Option.none() },
      owner: Option.some({ _tag: "Desktop", tray: { _tag: "Registered" } }),
    })).toBe([
      "Magnitude service",
      "  Runtime         Ready",
      "  Owner           Desktop",
      "  Tray            Registered",
      "  Starts at login Yes",
      "  Version         0.0.2",
      "  Address         127.0.0.1:10100",
      "  Active model    None",
      "",
    ].join("\n"))
  })
  it("does not present an unavailable model observation as an empty runtime", () => {
    const output = renderStatus({
      status: "Ready", address: "127.0.0.1:11101", version: Option.none(),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Unavailable" },
      owner: Option.some({ _tag: "Desktop", tray: { _tag: "Unavailable", message: "Desktop panel is unavailable" } }),
    })
    expect(output).toContain("Active model    Unavailable")
    expect(output).not.toContain("Active model    None")
    expect(output).toContain("Tray            Unavailable · Desktop panel is unavailable")
  })

  it.each(["Starting", "Ready", "Failed", "CleanupFailed"] as const)("reports headless %s without desktop fields", status => {
    const output = renderStatus({
      status, address: "127.0.0.1:11101", version: Option.none(),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Unavailable" },
      owner: Option.some({ _tag: "Headless" }),
    })
    expect(output).toContain(`Runtime         ${status}`)
    expect(output).toContain("Owner           Headless")
    expect(output).not.toMatch(/Tray|Starts at login/)
  })

})
