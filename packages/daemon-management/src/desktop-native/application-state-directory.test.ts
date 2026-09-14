import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { applicationStateDirectory } from "./application-state-directory"

const options = { platform: "win32" as const, dataDirectory: "D:\\Local Users\\模型\\.magnitude", override: Option.none<string>() }
describe("application ownership directory", () => {
  it("shares the user-data state directory on Windows", async () => {
    expect(await Effect.runPromise(applicationStateDirectory(options))).toBe("D:\\Local Users\\模型\\.magnitude\\state")
  })
  it("preserves extended local paths", async () => {
    expect(await Effect.runPromise(applicationStateDirectory({ ...options, dataDirectory: "\\\\?\\D:\\Users\\Local\\.magnitude" }))).toBe("\\\\?\\D:\\Users\\Local\\.magnitude\\state")
  })
  it("preserves explicit isolated profiles", async () => {
    expect(await Effect.runPromise(applicationStateDirectory({ ...options, override: Option.some("E:\\test\\state") }))).toBe("E:\\test\\state")
  })
  it.each(["darwin", "linux"] as const)("shares the %s service state root", async platform => {
    expect(await Effect.runPromise(applicationStateDirectory({ ...options, platform, dataDirectory: "/tmp/magnitude-profile" }))).toBe("/tmp/magnitude-profile/state")
  })
  it.each(["\\\\server\\share", "\\\\?\\UNC\\server\\share", "relative", "C:relative", "C:\\Users\0ignored"])("rejects unsupported user-data root %s without creating a second lock location", async dataDirectory => {
    expect((await Effect.runPromise(Effect.either(applicationStateDirectory({ ...options, dataDirectory }))))._tag).toBe("Left")
    expect((await Effect.runPromise(Effect.either(applicationStateDirectory({ ...options, override: Option.some(dataDirectory) }))))._tag).toBe("Left")
  })
})
