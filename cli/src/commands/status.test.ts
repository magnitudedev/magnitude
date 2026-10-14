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
      owner: Option.some({ _tag: "Desktop", tray: { _tag: "Registered" } }), reach: Option.none(), server: false, failing: Option.none(), loaded: Option.none(), elsewhere: Option.none(),
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
      owner: Option.some({ _tag: "Desktop", tray: { _tag: "Unavailable", message: "Desktop panel is unavailable" } }), reach: Option.none(), server: false, failing: Option.none(), loaded: Option.none(), elsewhere: Option.none(),
    })
    expect(output).toContain("Active model    Unavailable")
    expect(output).not.toContain("Active model    None")
    expect(output).toContain("Tray            Unavailable · Desktop panel is unavailable")
  })

  it.each(["Starting", "Ready", "Failed", "CleanupFailed"] as const)("reports headless %s without desktop fields", status => {
    const output = renderStatus({
      status, address: "127.0.0.1:11101", version: Option.none(),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Unavailable" },
      owner: Option.some({ _tag: "Headless" }), reach: Option.none(), server: false, failing: Option.none(), loaded: Option.none(), elsewhere: Option.none(),
    })
    expect(output).toContain(`Runtime         ${status}`)
    expect(output).toContain("Owner           Headless")
    expect(output).not.toMatch(/Tray|Starts at login/)
    expect(output).not.toContain("Open the Magnitude desktop app")
  })

  it("reports absence with actionable startup guidance and no desktop fields", () => {
    const output = renderStatus({ status: "Stopped", address: "127.0.0.1:10100", version: Option.none(),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Unavailable" }, owner: Option.none(), reach: Option.none(), server: false, failing: Option.none(), loaded: Option.none(), elsewhere: Option.none() })
    expect(output).toContain("Not running\nOpen the Magnitude desktop app or run `magnitude serve`.\n")
    expect(output).not.toMatch(/Tray|Starts at login|Active model/)
  })

  it.each([
    ["linux", "`sudo systemctl start magnitude`, and see its log: journalctl -u magnitude"],
    ["darwin", "`sudo launchctl bootstrap system /Library/LaunchDaemons/dev.magnitude.server.plist` (or, if launchd says it's already loaded, `sudo launchctl kickstart system/dev.magnitude.server`), and see its log: ~/.magnitude/logs/server.log"],
  ] as const)("points a stopped %s server at its service, not the desktop app or serve", (platform, advice) => {
    const output = renderStatus({ status: "Stopped", address: "127.0.0.1:10100", version: Option.none(),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Unavailable" }, owner: Option.none(), reach: Option.none(), server: true, failing: Option.none(), loaded: Option.none(), elsewhere: Option.none() }, platform)
    expect(output).toContain(`Not running\nThe Magnitude server isn't running. Start it with ${advice}\n`)
    expect(output).not.toMatch(/desktop app|magnitude serve/)
  })

  it("says a server systemd keeps restarting keeps failing, with its last error and log, instead of how to start it", () => {
    const output = renderStatus({ status: "Stopped", address: "127.0.0.1:10100", version: Option.none(),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Unavailable" }, owner: Option.none(), reach: Option.none(), server: true,
      failing: Option.some({ restarts: 4, lastError: Option.some("Magnitude couldn't download its inference engine (HTTP 404 from github.com).") }), loaded: Option.none(), elsewhere: Option.none() }, "linux")
    expect(output).toContain("Keeps failing to start\nThe Magnitude server keeps stopping as it starts (restarted 4 times).\n"
      + "Its last error: Magnitude couldn't download its inference engine (HTTP 404 from github.com).\nSee its log: journalctl -u magnitude\n")
    expect(output).not.toContain("systemctl start")
  })

  it.each([[true, "`sudo launchctl kickstart system/dev.magnitude.server`"], [false, "`sudo launchctl bootstrap system /Library/LaunchDaemons/dev.magnitude.server.plist`"]] as const)(
    "starts a macOS server launchd has loaded (%s) the way launchd accepts", (loaded, command) => {
      const output = renderStatus({ status: "Stopped", address: "127.0.0.1:10100", version: Option.none(), startsAutomaticallyOnLogin: Option.none(),
        activeModel: { _tag: "Unavailable" }, owner: Option.none(), reach: Option.none(), server: true, failing: Option.none(), loaded: Option.some(loaded), elsewhere: Option.none() }, "darwin")
      expect(output).toContain(`Start it with ${command}, and see its log`)
    })

  it("reports a server another account runs as running, not stopped", () => {
    const output = renderStatus({ status: "Ready", address: "127.0.0.1:10100", version: Option.some("50.8.7"), startsAutomaticallyOnLogin: Option.none(),
      activeModel: { _tag: "Unavailable" }, owner: Option.none(), reach: Option.none(), server: true, failing: Option.none(), loaded: Option.none(),
      elsewhere: Option.some({ account: Option.some("ec2-user") }) }, "darwin")
    expect(output).toContain("Runtime         Ready\n  Owner           Another account (ec2-user)\n  Version         50.8.7")
    expect(output).toContain("Magnitude is running on this computer for another account.")
    expect(output).not.toMatch(/Not running|kickstart|bootstrap/)
  })

  it("shows how to reach a ready server, without the key", () => {
    const output = renderStatus({
      status: "Ready", address: "127.0.0.1:10100", version: Option.some("1.0.0"),
      startsAutomaticallyOnLogin: Option.none(), activeModel: { _tag: "Observed", model: Option.none() },
      owner: Option.some({ _tag: "Headless" }),
      reach: Option.some({ endpoint: "http://127.0.0.1:10100", configPath: "/var/lib/magnitude/config.json", service: true,
        network: Option.some({ enabled: true, bind: Option.none(), port: 10100, pending: false, interfaces: [{ name: "eth0", address: "10.0.0.4", kind: "lan" }] }) }),
      server: true, failing: Option.none(), loaded: Option.none(), elsewhere: Option.none(),
    })
    expect(output).toContain("Address         0.0.0.0:10100 (all interfaces)")
    expect(output).toContain("Owner           Server")
    expect(output).toContain("On this computer:   http://127.0.0.1:10100")
    expect(output).toContain("http://10.0.0.4:10100")
    expect(output).toContain("http://10.0.0.4:10100/inference/v1")
  })
})
