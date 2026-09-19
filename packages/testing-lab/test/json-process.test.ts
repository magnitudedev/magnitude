import { expect, test } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { jsonProcess, type JsonProcess } from "../src/json-process"

const withProcess = <A>(source: string, run: (child: JsonProcess) => Effect.Effect<A, unknown>) =>
  Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-json-process-" })
    const child = yield* jsonProcess({ executable: process.execPath, args: ["-e", source], cwd: root, environment: {},
      stdoutLog: join(root, "stdout.jsonl"), stderrLog: join(root, "stderr.log") })
    yield* run(child)
  })).pipe(Effect.provide(BunContext.layer)))

test("JSONL preserves Unicode line separators and accepts CRLF", () => withProcess(
  'process.stdout.write(JSON.stringify({text:"first\\u2028second\\u2029third"})+"\\r\\n")',
  child => child.receive.pipe(Effect.tap(value => Effect.sync(() => expect(value).toEqual({ text: "first\u2028second\u2029third" }))))))

test.each([
  ['process.stdout.write("not JSON\\n")', "invalid JSONL"],
  ['process.stdout.write("{}")', "incomplete JSONL"],
  ['process.stdout.write(JSON.stringify("a".repeat(2*1024*1024))+"\\n")', "exceeded 2 MiB"],
])("rejects malformed or oversized output: %s", (source, message) => withProcess(source,
  child => child.receive.pipe(Effect.either, Effect.tap(result => Effect.sync(() => {
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toContain(message)
  })))) )

test("RPC writes are flushed and scoped shutdown terminates an idle process", () => withProcess(
  'for await (const line of console) { process.stdout.write(JSON.stringify({received: JSON.parse(line)})+"\\n") }',
  child => child.send({ id: "test", type: "state" }).pipe(Effect.zipRight(child.receive), Effect.tap(value => Effect.sync(() => {
    expect(value).toEqual({ received: { id: "test", type: "state" } })
  })))) )

test("scoped RPC shutdown escalates when the child ignores graceful termination", async () => {
  let pid = 0
  await withProcess('process.on("SIGTERM",()=>{});process.stdout.write(JSON.stringify({pid:process.pid})+"\\n");setInterval(()=>{},1000)',
    child => child.receive.pipe(Effect.flatMap(Schema.decodeUnknown(Schema.Struct({ pid: Schema.Int }))), Effect.tap(value => Effect.sync(() => { pid = value.pid }))))
  expect(pid).toBeGreaterThan(0)
  expect(() => process.kill(pid, 0)).toThrow()
}, 10000)
