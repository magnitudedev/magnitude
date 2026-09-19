import { expect, test } from "vitest"
import { Effect } from "effect"
import { command, ProcessExecutorLive } from "../src/process"

test("executes argv literally and captures both streams", async () => {
  const result = await Effect.runPromise(command(process.execPath, ["-e", "console.log(process.argv[1]); console.error('diagnostic')", "$(touch should-not-exist);'\nquoted"]).pipe(Effect.provide(ProcessExecutorLive)))
  expect(result.stdout).toBe("$(touch should-not-exist);'\nquoted\n")
  expect(result.stderr).toBe("diagnostic\n")
})
test("bounds output rather than accumulating forever", async () => {
  const result = await Effect.runPromise(command(process.execPath, ["-e", "console.log('x'.repeat(20000))"], { maxOutputBytes: 1000 }).pipe(Effect.either, Effect.provide(ProcessExecutorLive)))
  expect(result).toMatchObject({ _tag: "Left", left: { operation: "read-output" } })
})
test("times out and reaps long-lived subprocesses", async () => {
  const start = Date.now()
  const result = await Effect.runPromise(command(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { timeoutMs: 50 }).pipe(Effect.either, Effect.provide(ProcessExecutorLive)))
  expect(result).toMatchObject({ _tag: "Left", left: { operation: "timeout" } })
  expect(Date.now() - start).toBeLessThan(6000)
})

test("cleanup subprocess deadlines still apply inside an uninterruptible finalizer", async () => {
  const started = Date.now()
  await Effect.runPromise(Effect.scoped(Effect.addFinalizer(() => command(process.execPath,
    ["-e", "process.on('SIGTERM', () => {}); setInterval(() => {}, 1000)"], { timeoutMs: 250 })
    .pipe(Effect.either, Effect.tap(result => Effect.sync(() => {
      expect(result).toMatchObject({ _tag: "Left", left: { operation: "timeout" } })
    }))))).pipe(Effect.provide(ProcessExecutorLive)))
  expect(Date.now() - started).toBeLessThan(6000)
}, 8000)
