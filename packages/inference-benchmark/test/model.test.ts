import { describe, expect, it } from "vitest"
import { BunFileSystem } from "@effect/platform-bun"
import { Effect, Exit } from "effect"
import { resolveModelIdentity } from "../src/model"

describe("model identity", () => {
  it("requires a readable GGUF artifact", async () => {
    const exit = await Effect.runPromiseExit(
      resolveModelIdentity({
        id: "missing",
        artifactPath: "/definitely-not-a-model.gguf",
      }).pipe(Effect.provide(BunFileSystem.layer)),
    )
    expect(Exit.isFailure(exit)).toBe(true)
  })
})
