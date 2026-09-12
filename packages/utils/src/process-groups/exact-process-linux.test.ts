import { Effect, Either, Option } from "effect"
import { beforeEach, describe, expect, it, vi } from "vitest"

const { readFile } = vi.hoisted(() => ({ readFile: vi.fn() }))
vi.mock("node:fs/promises", () => ({ readFile }))

import { ProcessGroupControllerLive } from "./exact-process"

describe.skipIf(process.platform !== "linux")("Linux process disappearance", () => {
  beforeEach(() => readFile.mockReset())

  it.each(["ENOENT", "ESRCH"])("recognizes %s while reading process stat", async code => {
    readFile.mockRejectedValueOnce(Object.assign(new Error(code), { code }))
    const observed = await Effect.runPromise(ProcessGroupControllerLive.inspect(12345))
    expect(Option.isNone(observed)).toBe(true)
    expect(readFile).toHaveBeenCalledExactlyOnceWith("/proc/12345/stat", "utf8")
  })

  it.each(["EACCES", "EIO"])("preserves %s as an observation failure", async code => {
    readFile.mockRejectedValueOnce(Object.assign(new Error(code), { code }))
    const observed = await Effect.runPromise(Effect.either(ProcessGroupControllerLive.inspect(12345)))
    expect(Either.isLeft(observed)).toBe(true)
    if (Either.isLeft(observed)) expect(observed.left._tag).toBe("ExactProcessIdentityObservationFailed")
  })

  it("does not turn a boot identity read failure into process absence", async () => {
    readFile.mockResolvedValueOnce(`12345 (fixture) ${Array(20).fill("1").join(" ")}`)
    readFile.mockRejectedValueOnce(Object.assign(new Error("boot identity unavailable"), { code: "ESRCH" }))
    const observed = await Effect.runPromise(Effect.either(ProcessGroupControllerLive.inspect(12345)))
    expect(Either.isLeft(observed)).toBe(true)
    expect(readFile).toHaveBeenNthCalledWith(2, "/proc/sys/kernel/random/boot_id", "utf8")
  })
})
