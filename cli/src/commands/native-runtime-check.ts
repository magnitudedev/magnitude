import { Effect } from "effect"

/** Release-only headless probe: load the same FFI library as the terminal UI and exercise Bun. */
export const runNativeRuntimeCheck = () => Effect.runPromise(Effect.gen(function* () {
  const { resolveRenderLib } = yield* Effect.tryPromise(() => import("@opentui/core"))
  yield* Effect.sync(() => {
    const library = resolveRenderLib()
    const buffer = library.createTextBuffer("wcwidth")
    try {
      if (library.textBufferGetLength(buffer.ptr) !== 0) throw new Error("Native text buffer probe failed")
      const hot = (value: number) => (value * 31 + 7) | 0
      let value = 0
      for (let index = 0; index < 1_000_000; index++) value = hot(value)
      if (!Number.isInteger(value)) throw new Error("Bun runtime probe failed")
    } finally { buffer.destroy() }
    process.stdout.write("Bun and OpenTUI native runtime ready\n")
  })
}))
