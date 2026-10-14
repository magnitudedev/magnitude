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
      owner: Option.some({ _tag: "Desktop", tray: { _tag: "Registered" } }), reach: Option.none(), server: false,
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
      owner: Option.some({ _tag: "Desktop", tray: { _tag: "Unavailable", message: "Desktop panel is unavailable" } }), reach: Option.none(), server: false,
    })
    expect(output).toContain("Active model    Unavailable")
    expect(output).not.toContain("Active model    None")
    expect(output).toContain("Tray            Unavailable · Desktop panel is unavailable")
  })

  it.each(["Starting", "Ready", "Failed", "CleanupFailed"] as const)("reports headless %s without desktop fields", status => {
    const output = renderStatus({
      status, address: "127.0.0.1:11101", version: Option.none(),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Unavailable" },
      owner: Option.some({ _tag: "Headless" }), reach: Option.none(), server: false,
    })
    expect(output).toContain(`Runtime         ${status}`)
    expect(output).toContain("Owner           Headless")
    expect(output).not.toMatch(/Tray|Starts at login/)
    expect(output).not.toContain("Open the Magnitude desktop app")
  })

  it("reports absence with actionable startup guidance and no desktop fields", () => {
    const output = renderStatus({ status: "Stopped", address: "127.0.0.1:10100", version: Option.none(),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Unavailable" }, owner: Option.none(), reach: Option.none(), server: false })
    expect(output).toContain("Not running\nOpen the Magnitude desktop app or run `magnitude serve`.\n")
    expect(output).not.toMatch(/Tray|Starts at login|Active model/)
  })

  it.each([
    ["linux", "`sudo systemctl start magnitude`, and see its log: journalctl -u magnitude"],
    ["darwin", "`sudo launchctl kickstart system/dev.magnitude.server`, and see its log: ~/.magnitude/logs/server.log"],
  ] as const)("points a stopped %s server at its service, not the desktop app or serve", (platform, advice) => {
    const output = renderStatus({ status: "Stopped", address: "127.0.0.1:10100", version: Option.none(),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Unavailable" }, owner: Option.none(), reach: Option.none(), server: true }, platform)
    expect(output).toContain(`Not running\nThe Magnitude server isn't running. Start it with ${advice}\n`)
    expect(output).not.toMatch(/desktop app|magnitude serve/)
  })

  it("shows how to reach a ready server, without the key", () => {
    const output = renderStatus({
      status: "Ready", address: "127.0.0.1:10100", version: Option.some("1.0.0"),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Observed", model: Option.none() },
      owner: Option.some({ _tag: "Headless" }),
      reach: Option.some({ endpoint: "http://127.0.0.1:10100", configPath: "/var/lib/magnitude/config.json", service: true,
        network: Option.some({ enabled: true, port: 10100, pending: false, interfaces: [{ name: "eth0", address: "10.0.0.4", kind: "lan" }] }) }),
      server: true,
    })
    expect(output).toContain("Owner           Server")
    expect(output).toContain("On this computer:   http://127.0.0.1:10100")
    expect(output).toContain("http://10.0.0.4:10100")
    expect(output).toContain("http://10.0.0.4:10100/inference/v1")
  })
})
