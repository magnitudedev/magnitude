import { spawnSync } from "node:child_process"
import { describe, expect, it } from "vitest"
import { linuxRelaunchCommand } from "./linux-relaunch"

describe.skipIf(process.platform === "win32")("Linux relaunch", () => {
  it("starts the launcher with the arguments only once the given process has exited", () => {
    const { command, args } = linuxRelaunchCommand(999_999_999, "/bin/echo", ["--background", "a b"])
    const result = spawnSync(command, args, { encoding: "utf8" })
    expect(result.stdout).toBe("--background a b\n")
  })
})
