import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import { NativeHost, nativeHostLayerFromLoader } from "./index"

describe("privileged native adapter loading", () => {
  it.each([false, undefined, "true"])("refuses an unavailable or invalid desktop observation: %s", async value => {
    const layer = nativeHostLayerFromLoader(() => ({ isInteractiveDesktop: () => value }))
    const result = await Effect.runPromise(Effect.flatMap(NativeHost, native => native.requireInteractiveDesktop).pipe(Effect.provide(layer), Effect.either))
    expect(result._tag).toBe("Left")
  })
  it("accepts an interactive desktop without requiring it to be the unlocked input desktop", async () => {
    const layer = nativeHostLayerFromLoader(() => ({ isInteractiveDesktop: () => true }))
    await Effect.runPromise(Effect.flatMap(NativeHost, native => native.requireInteractiveDesktop).pipe(Effect.provide(layer)))
  })
  it("retains a native session-query failure", async () => {
    const layer = nativeHostLayerFromLoader(() => ({ isInteractiveDesktop: () => { throw new Error("access denied") } }))
    const result = await Effect.runPromise(Effect.flatMap(NativeHost, native => native.requireInteractiveDesktop).pipe(Effect.provide(layer), Effect.either))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toBe("Cannot inspect the Windows desktop session.")
  })
  it("is lazy and resolves the known folder from its supplied adapter", async () => {
    let loads = 0
    const layer = nativeHostLayerFromLoader(() => {
      loads++
      return { localAppDataDirectory: () => "C:\\Native Profile\\Local" }
    })
    expect(loads).toBe(0)
    const folder = await Effect.runPromise(Effect.flatMap(NativeHost, native => native.localAppDataDirectory).pipe(Effect.provide(layer)))
    expect(folder).toBe("C:\\Native Profile\\Local")
    expect(loads).toBe(1)
  })
  it("retains native loading failure instead of selecting an environment fallback", async () => {
    const layer = nativeHostLayerFromLoader(() => { throw new Error("invalid native installation") })
    const result = await Effect.runPromise(Effect.flatMap(NativeHost, native => native.localAppDataDirectory).pipe(Effect.provide(layer), Effect.either))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.message).toContain("invalid native installation")
  })
})
