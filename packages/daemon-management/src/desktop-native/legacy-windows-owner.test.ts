import { Effect, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { LegacyOwner } from "./legacy-owner"
import { legacyWindowsProcessIdentity } from "./legacy-windows-owner"

const owner = (processStartIdentity: string, pid = 42) => Schema.decodeUnknownSync(LegacyOwner)({
  pid, processStartIdentity, port: 10100,
})

describe("legacy Windows process identity", () => {
  it("converts the historical epoch without losing sub-millisecond precision", async () => {
    for (const [ticks, filetime] of [
      ["504911232000000000", "0000000000000000"],
      ["621355968000000000", "019db1ded53e8000"],
      ["621355968000000001", "019db1ded53e8001"],
    ]) {
      expect(await Effect.runPromise(legacyWindowsProcessIdentity(owner(`windows:${ticks}`))))
        .toEqual({ pid: 42, creationTime: filetime })
    }
  })

  it.each([
    "windows:1", "windows:504911231999999999", "windows:3155378976000000000",
    "windows:0621355968000000000", "windows:621355968000000000.0",
    "windows:6.21355968e17", "windows:+621355968000000000",
    "windows:621355968000000000\n", "linux:621355968000000000", "windows:not-a-number",
  ])("rejects malformed or out-of-range identity %s as a typed failure", async identity => {
    const result = await Effect.runPromise(Effect.either(legacyWindowsProcessIdentity(owner(identity))))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left._tag).toBe("LegacyWindowsIdentityInvalid")
  })

  it("does not admit a SQLite integer outside the native DWORD PID range", async () => {
    const result = await Effect.runPromise(Effect.either(legacyWindowsProcessIdentity(
      owner("windows:621355968000000000", 0x100000000),
    )))
    expect(result._tag).toBe("Left")
  })
})
