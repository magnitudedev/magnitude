import { Effect, Layer, Option, Schema } from "effect"
import { expect, test } from "vitest"
import { ProcessExecutor, type CommandSpec } from "../src/process"
import { linuxWorkerFault, linuxWorkerFaultScript, nativeWorkerFault, WorkerFault, WorkerFaultRequest } from "../src/worker-fault"

const request = Schema.decodeUnknownSync(WorkerFaultRequest)({ owner: { applicationPid: 10, servicePid: 11, serviceInstance: "owner" }, workerPid: 42, profile: "/private/profile with spaces; $(ignored)" })
const receipt = { workerPid: 42, parentPid: 20, parentStart: "123", executable: "/private/profile/releases/runtime/magnitude-inference", terminated: true }
test("native fault boundary passes literal scoped input and verifies the original parent", async () => {
  const calls: CommandSpec[] = []
  await Effect.runPromise(Effect.gen(function* () {
    const fault = yield* WorkerFault
    const actual = yield* fault.crash(request)
    yield* fault.verifyParent(actual)
  }).pipe(Effect.provide(linuxWorkerFault.pipe(Layer.provide(Layer.succeed(ProcessExecutor, { run: spec => {
    calls.push(spec)
    return Effect.succeed({ exitCode: 0, stdout: JSON.stringify(calls.length === 1 ? receipt : {}), stderr: "" })
  } }))))))
  expect(calls).toHaveLength(2)
  expect(calls[0]).toMatchObject({ executable: "/usr/bin/python3", args: ["-c", linuxWorkerFaultScript], inheritEnv: false, timeoutMs: 20000 })
  expect(JSON.parse(Option.getOrThrow(calls[0]!.stdin))).toEqual({ operation: "crash", request: Schema.encodeSync(WorkerFaultRequest)(request) })
  expect(JSON.parse(Option.getOrThrow(calls[1]!.stdin))).toEqual({ operation: "verify-parent", receipt })
})

test.each([{ exitCode: 1, stdout: "", stderr: "Not an owned worker" }, { exitCode: 0, stdout: "{}", stderr: "" },
  { exitCode: 0, stdout: JSON.stringify({ ...receipt, terminated: false }), stderr: "" }])("fault rejection or malformed receipt cannot pass: %j", async output => {
  const result = await Effect.runPromise(Effect.flatMap(WorkerFault, fault => fault.crash(request)).pipe(
    Effect.provide(linuxWorkerFault.pipe(Layer.provide(Layer.succeed(ProcessExecutor, { run: () => Effect.succeed(output) })))), Effect.either))
  expect(result._tag).toBe("Left")
})

test.skipIf(process.platform === "linux")("unqualified platform cannot invoke native termination", async () => {
  let executed = false
  const result = await Effect.runPromise(Effect.flatMap(WorkerFault, fault => fault.crash(request)).pipe(
    Effect.provide(nativeWorkerFault.pipe(Layer.provide(Layer.succeed(ProcessExecutor, { run: () => {
      executed = true
      return Effect.die("Must not execute")
    } })))), Effect.either))
  expect(result._tag).toBe("Left")
  expect(executed).toBe(false)
})
