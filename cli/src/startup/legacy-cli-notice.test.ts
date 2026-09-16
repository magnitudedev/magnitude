import { Effect, Fiber } from "effect"
import { afterEach, describe, expect, it, vi } from "vitest"
import { legacyCliNotice, showLegacyCliNotice } from "./legacy-cli-notice"

const streams = [process.stdin, process.stdout, process.stderr]
const descriptors = streams.map(stream => Object.getOwnPropertyDescriptor(stream, "isTTY"))
const terminal = (tty: boolean) => streams.forEach(stream =>
  Object.defineProperty(stream, "isTTY", { configurable: true, value: tty }))

describe("legacy CLI notice", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    streams.forEach((stream, index) => {
      const descriptor = descriptors[index]
      if (descriptor) Object.defineProperty(stream, "isTTY", descriptor)
      else Reflect.deleteProperty(stream, "isTTY")
    })
  })

  it("prints plain text without a delay when redirected", async () => {
    terminal(false)
    const write = vi.spyOn(process.stderr, "write").mockReturnValue(true)
    const start = Date.now()
    await Effect.runPromise(showLegacyCliNotice)
    expect(Date.now() - start).toBeLessThan(1000)
    expect(write).toHaveBeenCalledWith(`\n${legacyCliNotice}\n\n`)
    expect(legacyCliNotice).toContain("FREE, OPEN SOURCE")
    expect(legacyCliNotice).toContain("https://magnitude.dev")
  })

  it("prints before pausing and continues after three seconds", async () => {
    terminal(true)
    const write = vi.spyOn(process.stderr, "write").mockReturnValue(true)
    let continued = false
    const start = Date.now()
    const result = Effect.runPromise(showLegacyCliNotice.pipe(
      Effect.tap(() => Effect.sync(() => { continued = true })),
    ))
    expect(write).toHaveBeenCalledOnce()
    expect(continued).toBe(false)
    await result
    expect(continued).toBe(true)
    expect(Date.now() - start).toBeGreaterThanOrEqual(2900)
  })

  it("does not continue setup when the pause is interrupted", async () => {
    terminal(true)
    vi.spyOn(process.stderr, "write").mockReturnValue(true)
    let continued = false
    const fiber = Effect.runFork(showLegacyCliNotice.pipe(
      Effect.tap(() => Effect.sync(() => { continued = true })),
    ))
    await Effect.runPromise(Fiber.interrupt(fiber))
    expect(continued).toBe(false)
  })
})
