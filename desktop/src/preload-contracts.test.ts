import { describe, expect, it } from "vitest"
import { Effect } from "effect"
import { BunContext } from "@effect/platform-bun"
import { Command, CommandExecutor } from "@effect/platform"

// Electron's isolated preload world does not provide the renderer's Web Crypto API.
// Import in a fresh process so a renderer module cached by another test cannot hide this regression.
describe("preload contracts", () => {
  it("loads native RPC contracts without initializing renderer state", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const executor = yield* CommandExecutor.CommandExecutor
      return yield* executor.string(Command.make(process.execPath, "--eval", `
        Object.defineProperty(globalThis, "crypto", { value: {}, configurable: true });
        const { InferenceHostRpcs } = await import("./src/desktop-rpc.ts");
        if (!InferenceHostRpcs.requests.has("ApplicationInfo")) throw new Error("Missing native contract");
        console.log("preload contracts loaded");
      `))
    }).pipe(Effect.provide(BunContext.layer)))
    expect(result).toContain("preload contracts loaded")
  })
})
