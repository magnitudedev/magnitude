import { ConfigProvider, Effect } from "effect"
import { describe, expect, it } from "vitest"
import * as NodeContext from "@effect/platform-node/NodeContext"
import { isWindowsEngineLibrary, signWindowsCode, windowsSigning } from "./windows-signing"

describe("Windows distribution signing policy", () => {
  it("signs the common engine library while preserving Microsoft runtime signatures", () => {
    const runtime = ["ggml-base.dll", "ggml.dll", "llama-common.dll", "llama.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]
    expect(runtime.filter(isWindowsEngineLibrary)).toEqual(["ggml-base.dll", "ggml.dll", "llama-common.dll", "llama.dll"])
  })
  const config = (entries: readonly (readonly [string, string])[]) => ConfigProvider.fromMap(new Map(entries))
  it("allows local builds without invoking signing tools", async () => {
    await expect(Effect.runPromise(signWindowsCode("does-not-exist.exe").pipe(
      Effect.withConfigProvider(config([])),
      Effect.provide(NodeContext.layer),
    ))).resolves.toBeUndefined()
  })
  it("rejects a misspelled production mode instead of silently building unsigned", async () => {
    await expect(Effect.runPromise(Effect.gen(function* () { return yield* windowsSigning }).pipe(
      Effect.withConfigProvider(config([["MAGNITUDE_WINDOWS_DISTRIBUTION", "artifact-signng"]])),
    ))).rejects.toThrow()
  })
})
