import { Effect } from "effect"
import { afterEach, describe, expect, it, vi } from "vitest"
import { resolveServeOwner } from "./serve-owner.js"

describe("resolveServeOwner (systemd)", () => {
  const originalPlatform = process.platform

  afterEach(() => {
    vi.unstubAllEnvs()
    Object.defineProperty(process, "platform", { value: originalPlatform })
  })

  it("treats a foreground serve inheriting another systemd unit's environment as headless", async () => {
    Object.defineProperty(process, "platform", { value: "linux" })
    vi.stubEnv("INVOCATION_ID", "test-invocation")
    vi.stubEnv("SYSTEMD_EXEC_PID", String(process.ppid))

    expect(await Effect.runPromise(resolveServeOwner(Effect.succeed(false)))).toBe("headless")
  })

  it.each([String(process.pid), undefined])("recognizes systemd when SYSTEMD_EXEC_PID is %s", async executionPid => {
    Object.defineProperty(process, "platform", { value: "linux" })
    vi.stubEnv("INVOCATION_ID", "test-invocation")
    if (executionPid === undefined) {
      vi.stubEnv("SYSTEMD_EXEC_PID", undefined)
    } else {
      vi.stubEnv("SYSTEMD_EXEC_PID", executionPid)
    }

    expect(await Effect.runPromise(resolveServeOwner(Effect.succeed(false)))).toBe("service")
  })
})
