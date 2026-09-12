import { Effect, Fiber, Layer, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { WindowsPrivatePipes, WindowsProcessId, windowsJobOwnerLayer } from "@magnitudedev/utils/windows-native"
import { makeWindowsLegacyTaskControl } from "./windows-task-query"
import { LegacyWindowsStartup } from "./legacy-startup-windows"

const fixture = (chunks: readonly Uint8Array[], initialExit: number | null = 0, stalled = false, retirement?: LegacyWindowsStartup) => {
  let exit = initialExit
  let offset = 0
  const state = { closed: 0, killed: 0, pipeClosed: 0, executable: "", command: "" }
  const jobs = windowsJobOwnerLayer({
    spawnOwnedProcess: (executable, command) => { state.executable = executable; state.command = command; return {} },
    spawnOwnedProcessWithPipes: () => { throw new Error("Unexpected separate streams") },
    ownedProcessIdentity: () => ({ pid: 42, creationTime: "1234567890abcdef" }),
    ownedProcessActiveCount: () => exit === null ? 1 : 0,
    ownedProcessExit: () => exit,
    terminateOwnedProcess: () => { state.killed++; if (exit === null) exit = 1 },
    closeOwnedProcess: () => { state.closed++ },
  })
  const pipes = Layer.succeed(WindowsPrivatePipes, WindowsPrivatePipes.of({ bind: () => Effect.gen(function* () {
    const close = Effect.sync(() => { state.pipeClosed++ })
    yield* Effect.addFinalizer(() => close)
    return {
      accept: Effect.succeed(WindowsProcessId.make(process.pid)),
      read: stalled ? Effect.never : Effect.sync(() => chunks[offset++] ?? new Uint8Array()),
      write: () => Effect.void, close,
    }
  }) }))
  const run = Effect.scoped(Effect.gen(function* () {
    const query = yield* makeWindowsLegacyTaskControl({ executable: "C:\\Magnitude App\\magnitude-task-query.exe", environment: {}, timeout: "30 millis" })
    return yield* (retirement ? query.retire(retirement) : query.query)
  })).pipe(Effect.provide(Layer.merge(jobs, pipes)))
  return { run, state }
}
describe("Windows task control (simulated native boundary)", () => {
  it("reads fragmented UTF-8 and proves retirement before returning a snapshot", async () => {
    const reply = { _tag: "Registered", xml: '<Task><Description>模型🙂</Description></Task>', currentUserSid: "S-1-5-21-123-456-789-1001" }
    const test = fixture([...Buffer.from(JSON.stringify(reply))].map(byte => Uint8Array.of(byte)))
    expect(await Effect.runPromise(test.run)).toEqual(reply)
    expect(test.state.closed).toBe(1)
    expect(test.state.killed).toBe(1)
    expect(test.state.pipeClosed).toBe(1)
    expect(test.state.command).toBe('"C:\\Magnitude App\\magnitude-task-query.exe"')
  })
  it("accepts only explicit task absence", async () => {
    const test = fixture([Buffer.from('{"_tag":"Missing"}\n')])
    expect(await Effect.runPromise(test.run)).toEqual({ _tag: "Missing" })
  })
  const registration = Schema.decodeUnknownSync(LegacyWindowsStartup)({
    _tag: "WindowsScheduledTask", task: "\\MagnitudeInference", digest: "a".repeat(64),
    enabled: false, executable: "C:\\Magnitude App\\magnitude-service.exe", principalSid: "S-1-5-21-1",
  })
  it("retires only the captured digest and user, accepting already-absent replay", async () => {
    const test = fixture([Buffer.from('{"_tag":"Missing"}')], 0, false, registration)
    expect(await Effect.runPromise(test.run)).toBeUndefined()
    expect(test.state.command).toBe(`"C:\\Magnitude App\\magnitude-task-query.exe" "--retire" "${registration.digest}" "${registration.principalSid}"`)
    expect(test.state.closed).toBe(1)
    expect(test.state.pipeClosed).toBe(1)
  })
  it.each([
    [{ _tag: "Registered", xml: "<Task/>", currentUserSid: "S-1-5-21-1" }, 0, "remains registered"],
    [{ _tag: "Failed", hresult: 0x8007051a }, 1, "0x8007051a"],
  ] as const)("does not acknowledge unproven task retirement %#", async (reply, code, message) => {
    const test = fixture([Buffer.from(JSON.stringify(reply))], code, false, registration)
    const result = await Effect.runPromise(Effect.either(test.run))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toContain(message)
    expect(test.state.closed).toBe(1)
  })
  it.each([
    [Buffer.from('{"_tag":"Failed","hresult":2147942405}'), 1, "0x80070005"],
    [Buffer.from('{"_tag":"Missing"}'), 3, "code 3"],
    [Buffer.from('{"_tag":"Missing","extra":true}'), 0, "invalid snapshot"],
    [Uint8Array.of(0xff), 0, "invalid UTF-8"],
    [Buffer.alloc(512 * 1024 + 1), 0, "output limit"],
  ] as const)("rejects failed or invalid query output %#", async (bytes, code, message) => {
    const test = fixture([bytes], code)
    const result = await Effect.runPromise(Effect.either(test.run))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toContain(message)
    expect(test.state.closed).toBe(1)
    expect(test.state.pipeClosed).toBe(1)
  })
  it("bounds stalled COM/output and retires the helper job", async () => {
    const test = fixture([], null, true)
    const result = await Effect.runPromise(Effect.either(test.run))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toContain("timed out")
    expect(test.state.killed).toBe(1)
    expect(test.state.closed).toBe(1)
    expect(test.state.pipeClosed).toBe(1)
  })
  it("cancels a pending query without leaving its helper running", async () => {
    const test = fixture([], null, true)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fiber = yield* Effect.forkScoped(test.run)
      while (test.state.executable === "") yield* Effect.sleep("1 millis")
      yield* Fiber.interrupt(fiber)
    })))
    expect(test.state.killed).toBe(1)
    expect(test.state.closed).toBe(1)
    expect(test.state.pipeClosed).toBe(1)
  })
})
