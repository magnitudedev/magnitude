import { describe, expect, it } from "vitest"
import { admitsPkexecCommand } from "./pkexec-admission"

describe("pkexec command admission", () => {
  const pkexec = { PKEXEC_UID: "1000" }
  it.each([
    [["_install-application-update", "/home/ada/.magnitude/updates/update.json"]],
    [["_server-remove"]],
  ])("admits %j as root under pkexec", argv => {
    expect(admitsPkexecCommand(argv, 0, pkexec)).toBe(true)
  })
  it.each([
    [[]],
    [["_server-install"]],
    [["serve"]],
    [["status"]],
    [["update"]],
    [["_complete-application-update"]],
    [["-v", "_install-application-update", "/a/updates/update.json"]],
    [["--help"]],
  ])("refuses %j as root under pkexec", argv => {
    expect(admitsPkexecCommand(argv, 0, pkexec)).toBe(false)
  })
  it("admits every command without pkexec or root", () => {
    expect(admitsPkexecCommand(["serve"], 0, {})).toBe(true)
    expect(admitsPkexecCommand(["serve"], 0, { SUDO_UID: "1000" })).toBe(true)
    expect(admitsPkexecCommand(["serve"], 1000, pkexec)).toBe(true)
  })
})
