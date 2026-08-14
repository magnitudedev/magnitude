import { afterEach, describe, expect, it, vi } from "vitest"
import { Effect } from "effect"
import {
  flushProcessOutput,
  scheduleBoundedProcessExit,
  type FlushableOutput,
} from "./flush-process-output"

afterEach(() => {
  vi.useRealTimers()
})

describe("flushProcessOutput", () => {
  it("waits for stdout and stderr write callbacks", async () => {
    vi.useFakeTimers()
    const write = vi.fn((_chunk: string, callback: (error?: Error | null) => void) => {
      callback()
      return true
    })
    const stream: FlushableOutput = { write }

    await Effect.runPromise(flushProcessOutput({ stdout: stream, stderr: stream, timeoutMs: 25 }))

    expect(write).toHaveBeenCalledTimes(2)
    expect(vi.getTimerCount()).toBe(0)
  })

  it("stops waiting when output streams do not drain", async () => {
    const blocked: FlushableOutput = {
      write: vi.fn(() => false),
    }

    await Effect.runPromise(flushProcessOutput({ stdout: blocked, stderr: blocked, timeoutMs: 5 }))

    expect(blocked.write).toHaveBeenCalledTimes(2)
  })

  it("allows a bounded late-cleanup grace before forcing the requested status", () => {
    const previousExitCode = process.exitCode
    const exit = vi.spyOn(process, "exit").mockImplementation((() => undefined) as never)
    const unref = vi.fn()
    let forceExit: (() => void) | undefined
    const setTimeout = vi.spyOn(globalThis, "setTimeout").mockImplementation(((
      callback: () => void,
    ) => {
      forceExit = callback
      return { unref }
    }) as unknown as typeof globalThis.setTimeout)

    try {
      Effect.runSync(scheduleBoundedProcessExit(143, 25))

      expect(process.exitCode).toBe(143)
      expect(setTimeout).toHaveBeenCalledWith(expect.any(Function), 25)
      expect(unref).toHaveBeenCalledTimes(1)
      expect(exit).not.toHaveBeenCalled()
      forceExit?.()
      expect(exit).toHaveBeenCalledWith(143)
    } finally {
      setTimeout.mockRestore()
      exit.mockRestore()
      process.exitCode = previousExitCode ?? 0
    }
  })
})
