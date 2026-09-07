import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  WINDOWS_SERVICE_ENABLED_SCRIPT,
  WINDOWS_SERVICE_PID_SCRIPT,
  parseWindowsServicePid,
} from "./service"

describe("Windows service process detection", () => {
  it("accepts a positive task process ID", () => {
    expect(parseWindowsServicePid(" 4128\r\n")).toEqual(Option.some(4128))
  })

  it("rejects missing and non-positive process IDs", () => {
    expect(parseWindowsServicePid("")).toEqual(Option.none())
    expect(parseWindowsServicePid("0")).toEqual(Option.none())
    expect(parseWindowsServicePid("-1")).toEqual(Option.none())
  })

  it("uses the scheduled task action and Win32 process API", () => {
    expect(WINDOWS_SERVICE_PID_SCRIPT).toContain("Get-ScheduledTask")
    expect(WINDOWS_SERVICE_PID_SCRIPT).toContain("Get-CimInstance Win32_Process")
    expect(WINDOWS_SERVICE_ENABLED_SCRIPT).toContain("Get-ScheduledTask")
    expect(WINDOWS_SERVICE_ENABLED_SCRIPT).toContain("Disabled")
  })
})
