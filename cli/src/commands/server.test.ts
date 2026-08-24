import { describe, expect, it } from "vitest"
import {
  renderLinuxServerService,
  renderMacServerService,
  renderWindowsServerCommand,
  WINDOWS_RESTART_POLICY_SCRIPT,
} from "./server"

describe("Magnitude server service definitions", () => {
  it("renders a launch agent with exact argv and automatic restart", () => {
    const rendered = renderMacServerService(["/Applications/Magnitude & Tools/acn", "server"])
    expect(rendered).toContain("<string>/Applications/Magnitude &amp; Tools/acn</string><string>server</string>")
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
    expect(renderWindowsServerCommand([
      "C:\\Users\\Magnitude User\\magnitude-acn.exe",
      "serve",
      "a\\\"b",
      "trailing \\",
    ])).toBe('"C:\\Users\\Magnitude User\\magnitude-acn.exe" serve "a\\\\\\\"b" "trailing \\\\"')
    expect(WINDOWS_RESTART_POLICY_SCRIPT).toContain("ExecutionTimeLimit = 'PT0S'")
    expect(WINDOWS_RESTART_POLICY_SCRIPT).toContain("RestartCount = 999")
    expect(WINDOWS_RESTART_POLICY_SCRIPT).toContain("RestartInterval = 'PT1M'")
  })
})
