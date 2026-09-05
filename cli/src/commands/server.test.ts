import { describe, expect, it } from "vitest"
import { Command } from "@commander-js/extra-typings"
import { ACN_EXECUTABLE_NAME } from "@magnitudedev/daemon-management"
import { Option } from "effect"
import {
  renderLinuxServerService,
  renderMacServerService,
  renderWindowsServerCommand,
  WINDOWS_RESTART_POLICY_SCRIPT,
} from "@magnitudedev/daemon-management/service"
import { developmentServerCommand } from "../server/service"
import { registerServiceCommand } from "./server"
import { renderServiceStatus } from "./server-runtime"

describe("Magnitude service definitions", () => {
  it("registers only the public service command group", () => {
    const program = new Command().name("magnitude")
    registerServiceCommand(program)
    expect(program.commands.map((command) => command.name())).toEqual(["service"])
    expect(program.commands[0]!.commands.map((command) => command.name())).toEqual([
      "install",
      "uninstall",
      "start",
      "stop",
      "status",
    ])
  })
  it("registers development startup against the local ACN entrypoint", () => {
    expect(developmentServerCommand("/opt/bun")).toEqual([
      "/opt/bun",
      expect.stringMatching(/packages\/acn\/src\/binary\.ts$/),
      "serve",
    ])
  })

  it("renders a launch agent with exact argv and automatic restart", () => {
    const executable = `/Applications/Magnitude & Tools/${ACN_EXECUTABLE_NAME}`
    const rendered = renderMacServerService([executable, "server"])
    expect(rendered).toContain(
      `<string>/Applications/Magnitude &amp; Tools/${ACN_EXECUTABLE_NAME}</string><string>server</string>`,
    )
    expect(rendered).toContain("<key>RunAtLoad</key><true/>")
    expect(rendered).toContain("<key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>")
  })

  it("renders a user service with shell-safe argv and restart policy", () => {
    const rendered = renderLinuxServerService(["/opt/Magnitude%20Tools/acn", "it's", "server"])
    expect(rendered).toContain('ExecStart="/opt/Magnitude%%20Tools/acn" "it\'s" "server"')
    expect(rendered).toContain("Restart=on-failure")
    expect(rendered).toContain("WantedBy=default.target")
  })

  it("renders a Windows task command with native argv quoting and restart policy", () => {
    const executable = `C:\\Users\\Magnitude User\\${ACN_EXECUTABLE_NAME}.exe`
    expect(renderWindowsServerCommand([
      executable,
      "serve",
      "a\\\"b",
      "trailing \\",
    ])).toBe(`"${executable}" serve "a\\\\\\\"b" "trailing \\\\"`)
    expect(WINDOWS_RESTART_POLICY_SCRIPT).toContain("ExecutionTimeLimit = 'PT0S'")
    expect(WINDOWS_RESTART_POLICY_SCRIPT).toContain("RestartCount = 999")
    expect(WINDOWS_RESTART_POLICY_SCRIPT).toContain("RestartInterval = 'PT1M'")
  })

  it("renders service status as a labeled product summary", () => {
    expect(renderServiceStatus({
      status: "Ready",
      address: "127.0.0.1:10100",
      version: Option.some("0.0.2"),
      startsAutomaticallyOnLogin: true,
      activeModel: Option.none(),
    })).toBe([
      "Magnitude service",
      "  Runtime         Ready",
      "  Starts at login Yes",
      "  Version         0.0.2",
      "  Address         127.0.0.1:10100",
      "  Active model    None",
      "",
    ].join("\n"))
  })
})
