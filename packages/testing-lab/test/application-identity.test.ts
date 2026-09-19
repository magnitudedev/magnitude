import { Effect, Schema } from "effect"
import { spawn } from "node:child_process"
import { expect, test } from "vitest"
import { assertServiceExited, LabProcessId, ReadyApplicationSnapshot } from "../src/application-identity"

test("refuses a live process without terminating it and accepts an exited child", async () => {
  expect((await Effect.runPromise(assertServiceExited(LabProcessId.make(process.pid)).pipe(Effect.either)))._tag).toBe("Left")
  expect(() => process.kill(process.pid, 0)).not.toThrow()
  const child = spawn(process.execPath, ["-e", "process.exit(0)"], { stdio: "ignore" })
  await new Promise<void>((resolve, reject) => {
    child.once("error", reject)
    child.once("exit", code => code === 0 ? resolve() : reject(new Error(`Child exited ${code}`)))
  })
  expect(child.pid).toBeDefined()
  await Effect.runPromise(assertServiceExited(LabProcessId.make(child.pid!)))
})
test("requires an actual ready service identity in the native observation", () => {
  const decode = Schema.decodeUnknownEither(ReadyApplicationSnapshot)
  expect(decode({ pid: 1, endpoint: "http://127.0.0.1:11399", service: { _tag: "Starting" } })._tag).toBe("Left")
  expect(decode({ pid: 1, endpoint: "http://127.0.0.1:11399", service: { _tag: "Ready", health: { pid: 0, id: "" } } })._tag).toBe("Left")
})
