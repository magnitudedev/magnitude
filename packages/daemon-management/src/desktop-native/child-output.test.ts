import { Writable } from "node:stream"
import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { makeChildOutput } from "./child-output"

describe("owned child diagnostics", () => {
  it("retains the last 16 KiB without writing desktop diagnostics to the terminal", async () => {
    const sink = new Writable({ write() { throw new Error("unexpected terminal output") } })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const output = yield* makeChildOutput("DiagnosticTail", sink)
      yield* output.append(Buffer.alloc(1_000_000, "a"))
      yield* output.append(Buffer.from("last output"))
      const tail = yield* output.diagnosticTail
      expect(Buffer.byteLength(tail)).toBe(16_384)
      expect(tail.endsWith("last output")).toBe(true)
    })))
  })

  it("bounds a blocked terminal write and closes scope without waiting on the terminal", async () => {
    let writes = 0
    const sink = new Writable({ write() { writes++ } })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const output = yield* makeChildOutput("Foreground", sink)
      for (let index = 0; index < 100; index++) yield* output.append(Buffer.alloc(100_000, "a"))
      yield* output.append(Buffer.from("final diagnostic"))
      expect(writes).toBe(1)
      expect(sink.writableLength).toBe(16_384)
      expect((yield* output.diagnosticTail).endsWith("final diagnostic")).toBe(true)
    })).pipe(Effect.timeout("1 second")))
    expect(sink.destroyed).toBe(false)
    sink.destroy()
  })

  it("survives a terminal write failure after scope closure", async () => {
    let complete: ((error?: Error | null) => void) | undefined
    const sink = new Writable({ write(_chunk, _encoding, callback) { complete = callback } })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const output = yield* makeChildOutput("Foreground", sink)
      yield* output.append(Buffer.from("diagnostic"))
    })))
    complete!(new Error("terminal closed"))
    await new Promise<void>(resolve => setImmediate(resolve))
    expect(sink.listenerCount("error")).toBe(0)
  })

  it("keeps collecting after a synchronous sink failure", async () => {
    const sink = new Writable({ write() { throw new Error("terminal unavailable") } })
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const output = yield* makeChildOutput("Foreground", sink)
      yield* output.append(Buffer.from("first"))
      yield* output.append(Buffer.from("second"))
      expect(yield* output.diagnosticTail).toBe("firstsecond")
    })))
    expect(sink.listenerCount("error")).toBe(0)
  })
})
