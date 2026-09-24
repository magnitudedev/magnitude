import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { decodeMacInstallerInvocation } from "./mac-installer-command"
const request = { protocol: 1, operation: "Install", bundle: "/Applications/Magnitude.app", stateDirectory: "/Users/test/.magnitude/state",
  dataDirectory: "/Users/test/.magnitude", continuation: { _tag: "Foreground", arguments: ["serve", "--log-level", "debug"] } }
const executable = "/Users/test/.magnitude/state/mac-installers/installer-Abc123/magnitude"
const decode = (value: unknown, path = executable, descriptor: string | undefined = "12") =>
  decodeMacInstallerInvocation(JSON.stringify(value), path, descriptor)
describe("macOS private installer invocation", () => {
  it("binds a foreground invocation to the executing private helper and inherited descriptor", async () => {
    const result = await Effect.runPromise(decode(request))
    expect(result.request).toEqual(request)
    expect(result.descriptor).toBe(12)
    expect(result.directory).toBe(executable.slice(0, -"/magnitude".length))
  })
  it("accepts finite installation without startup continuation", async () => {
    expect((await Effect.runPromise(decode({ ...request, continuation: { _tag: "None" } }))).request.continuation._tag).toBe("None")
  })
  it("accepts recovery as distinct from another installation attempt", async () => {
    expect((await Effect.runPromise(decode({ ...request, operation: "Recover" }))).request.operation).toBe("Recover")
  })
  it.each([undefined, "0", "2", "-1", "3.5", "1e2", "03", "2147483648", "12suffix"])("rejects descriptor %s", async descriptor => {
    expect(await Effect.runPromise(decodeMacInstallerInvocation(JSON.stringify(request), executable, descriptor).pipe(Effect.isFailure))).toBe(true)
  })
  it.each([
    { ...request, protocol: 2 }, { ...request, operation: "RetryAutomatically" }, { ...request, operation: undefined },
    { ...request, bundle: "/Applications/../Magnitude.app" },
    { ...request, dataDirectory: "relative" }, { ...request, unknown: true },
    { ...request, continuation: { _tag: "Foreground", arguments: ["app", "open"] } },
    { ...request, continuation: { _tag: "Foreground", arguments: ["serve", "bad\0argument"] } },
    { ...request, continuation: { _tag: "Foreground", arguments: [] } },
  ])("rejects malformed request %#", async value => {
    expect(await Effect.runPromise(decode(value).pipe(Effect.isFailure))).toBe(true)
  })
  it.each(["/Applications/Magnitude.app/Contents/Resources/magnitude", executable.replace("installer-Abc123", "unrelated"),
    executable.replace("/Users/test/", "/Users/other/"), executable + "/../magnitude"])("rejects unrelated executing path %s", async path => {
    expect(await Effect.runPromise(decode(request, path).pipe(Effect.isFailure))).toBe(true)
  })
  it("rejects oversized payload before decoding", async () => {
    expect(await Effect.runPromise(decodeMacInstallerInvocation(" ".repeat(65537), executable, "12").pipe(Effect.isFailure))).toBe(true)
  })
})
