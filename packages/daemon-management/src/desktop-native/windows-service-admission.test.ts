import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { LegacyOwner, LegacyOwnerReadFailed } from "./legacy-owner"
import { LegacyStartupFailed } from "./legacy-startup-command"
import { LegacyWindowsStartup } from "./legacy-startup-windows"
import { requireFreshWindowsInstallation } from "./windows-service-admission"

const missing = Effect.succeed(Option.none<LegacyWindowsStartup>())
const owner = Schema.decodeUnknownSync(LegacyOwner)({ pid: 42, processStartIdentity: "windows:621355968000000000", port: 1234 })

describe("Windows service admission", () => {
  it("admits only independently observed task and record absence", async () => {
    await Effect.runPromise(requireFreshWindowsInstallation({ owner: Effect.succeed(Option.none()), task: missing }))
  })
  it("retains a recorded predecessor even when the task is absent", async () => {
    const result = await Effect.runPromise(requireFreshWindowsInstallation({ owner: Effect.succeed(Option.some(owner)), task: missing }).pipe(Effect.either))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toContain("requires migration")
  })
  it("does not treat a registered task without an owner record as absence", async () => {
    const task = Schema.decodeUnknownSync(LegacyWindowsStartup)({ _tag: "WindowsScheduledTask", task: "\\MagnitudeInference", digest: "a".repeat(64), enabled: true, executable: "C:\\Magnitude\\magnitude-service.exe", principalSid: "S-1-5-21-1" })
    const result = await Effect.runPromise(requireFreshWindowsInstallation({ owner: Effect.succeed(Option.none()), task: Effect.succeed(Option.some(task)) }).pipe(Effect.either))
    expect(result._tag).toBe("Left")
  })
  it("rejects malformed historical identity before querying the scheduler", async () => {
    let queries = 0
    const result = await Effect.runPromise(requireFreshWindowsInstallation({
      owner: Effect.succeed(Option.some({ ...owner, processStartIdentity: LegacyOwner.fields.processStartIdentity.make("windows:1") })),
      task: Effect.sync(() => { queries++; return Option.none<LegacyWindowsStartup>() }),
    }).pipe(Effect.either))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toContain("UTC process-start identity")
    expect(queries).toBe(0)
  })
  it("retains database and scheduler failures instead of launching", async () => {
    for (const observations of [
      { owner: Effect.fail(new LegacyOwnerReadFailed({ path: "fixture", message: "database denied" })), task: missing },
      { owner: Effect.succeed(Option.none<LegacyOwner>()), task: Effect.fail(new LegacyStartupFailed({ message: "scheduler unavailable" })) },
    ]) {
      const result = await Effect.runPromise(requireFreshWindowsInstallation(observations).pipe(Effect.either))
      expect(result._tag).toBe("Left")
      if (result._tag === "Left") expect(result.left.message).toMatch(/denied|unavailable/)
    }
  })
})
