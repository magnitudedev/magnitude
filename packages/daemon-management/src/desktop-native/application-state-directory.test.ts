import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { applicationStateDirectory } from "./application-state-directory"
import { NativeHostUnavailable } from "./index"

const unavailable = Effect.fail(new NativeHostUnavailable({ message: "Known folder unavailable" }))
const options = { platform: "win32" as const, dataDirectory: "\\\\server\\roaming\\home", development: false, override: Option.none<string>(), localAppDataDirectory: Effect.succeed("D:\\Local Users\\模型") }
describe("application ownership directory", () => {
  it("uses the native local folder independently of redirected home and separates development", async () => {
    expect(await Effect.runPromise(applicationStateDirectory(options))).toBe("D:\\Local Users\\模型\\Magnitude\\desktop")
    expect(await Effect.runPromise(applicationStateDirectory({ ...options, development: true }))).toBe("D:\\Local Users\\模型\\Magnitude Development\\desktop")
  })
  it("preserves extended local paths", async () => {
    expect(await Effect.runPromise(applicationStateDirectory({ ...options, localAppDataDirectory: Effect.succeed("\\\\?\\D:\\Users\\Local") }))).toBe("\\\\?\\D:\\Users\\Local\\Magnitude\\desktop")
  })
  it("does not query native folders for an explicit isolated override", async () => {
    expect(await Effect.runPromise(applicationStateDirectory({ ...options, override: Option.some("E:\\test\\desktop"), localAppDataDirectory: unavailable }))).toBe("E:\\test\\desktop")
  })
  it.each(["darwin", "linux"] as const)("preserves %s data-relative placement without Windows discovery", async platform => {
    expect(await Effect.runPromise(applicationStateDirectory({ ...options, platform, dataDirectory: "/tmp/magnitude-profile", localAppDataDirectory: unavailable }))).toBe("/tmp/magnitude-profile/desktop")
  })
  it.each(["\\\\server\\share", "relative", "C:relative", "C:\\Users\0ignored"])("rejects unsupported native folder %s", async folder => {
    const result = await Effect.runPromise(Effect.either(applicationStateDirectory({ ...options, localAppDataDirectory: Effect.succeed(folder) })))
    expect(result._tag).toBe("Left")
  })
  it("propagates native lookup failure without guessing a folder", async () => {
    const result = await Effect.runPromise(Effect.either(applicationStateDirectory({ ...options, localAppDataDirectory: unavailable })))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toBe("Known folder unavailable")
  })
})
