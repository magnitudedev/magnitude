import { describe, expect, it } from "vitest"
import { Cause, Effect, Option } from "effect"
import { CatalogFormModelIdSchema } from "@magnitudedev/acn-protocol"
import { catalogRemovalOutcome, type LocalModelRemovalsApi } from "./local-model-removals"

const modelId = CatalogFormModelIdSchema.make("qwen3.5-4b:gguf:q4")

describe("catalog removal outcomes", () => {
  it("accepts an actual removal", async () => {
    await expect(Effect.runPromise(catalogRemovalOutcome(modelId)({
      _tag: "Removed",
      reclaimedBytes: 42,
    }))).resolves.toEqual({})
  })

  it.each([
    ["ExternalOwnership" as const, "model_removal_retained_external"],
    ["SharedMaterial" as const, "model_removal_retained_shared"],
  ])("does not report a retained installation as removed", async (reason, code) => {
    const exit = await Effect.runPromiseExit(catalogRemovalOutcome(modelId)({ _tag: "Retained", reason }))
    expect(exit._tag).toBe("Failure")
    if (exit._tag === "Failure") {
      const failure = Option.getOrThrow(Cause.failureOption(exit.cause))
      expect(failure.code).toBe(code)
      if (reason === "ExternalOwnership") expect(failure.message).toContain(`magnitude models hide ${modelId}`)
      else expect(failure.message).not.toContain("magnitude models hide")
    }
  })
})

describe("catalog removal state", () => {
  const removalsWith = async (removeResult: Effect.Effect<unknown, unknown>) => {
    const { IcnClient, IcnCatalog, IcnLocalModels } = await import("@magnitudedev/icn")
    const { LocalModelRemovals, LocalModelRemovalsLive } = await import("./local-model-removals")
    return <A, E>(use: (removals: LocalModelRemovalsApi) => Effect.Effect<A, E>) => Effect.runPromise(
      LocalModelRemovals.pipe(
        Effect.flatMap(use),
        Effect.provide(LocalModelRemovalsLive),
        Effect.provideService(IcnClient, { catalog: { removeCatalogModelInstallation: () => removeResult } } as never),
        Effect.provideService(IcnCatalog, { refresh: Effect.void } as never),
        Effect.provideService(IcnLocalModels, { pruneDrafters: Effect.void } as never),
      ),
    )
  }
  const remoteError = (code: string, message: string) => Effect.fail({
    _tag: "GeneratedClientRemoteError",
    operationId: "removeCatalogModelInstallation",
    status: 409,
    headers: {},
    body: { error: { code, message, type: "model_error", param: Option.none() } },
  })

  it("answers a removal of a model that isn't installed clearly and leaves no failed state", async () => {
    const run = await removalsWith(remoteError("catalog_model_not_installed", `catalog model ${modelId} is not installed`))
    const { failure, state } = await run((removals) => Effect.gen(function* () {
      const failure = yield* Effect.flip(removals.remove(modelId))
      return { failure, state: yield* removals.state }
    }))
    expect(failure).toMatchObject({ _tag: "LocalModelMutationFailed", code: "model_not_installed" })
    expect(failure.message).toBe(`${modelId} is not installed on this computer, so there is nothing to remove.`)
    expect(state.size).toBe(0)
  })

  it("answers a model whose weights only another app holds with Hide and leaves no failed state", async () => {
    const run = await removalsWith(Effect.succeed({ _tag: "Retained", reason: "ExternalOwnership" }))
    const { failure, state } = await run((removals) => Effect.gen(function* () {
      const failure = yield* Effect.flip(removals.remove(modelId))
      return { failure, state: yield* removals.state }
    }))
    expect(failure).toMatchObject({ _tag: "LocalModelMutationFailed", code: "model_removal_retained_external" })
    expect(failure.message).toContain(`magnitude models hide ${modelId}`)
    expect(state.size).toBe(0)
  })

  it("keeps a real failure until a new installation supersedes it", async () => {
    const run = await removalsWith(remoteError("remove_failed", "Removal failed"))
    const { failed, forgotten } = await run((removals) => Effect.gen(function* () {
      yield* Effect.flip(removals.remove(modelId))
      const failed = (yield* removals.state).get(modelId)
      yield* removals.forget(modelId)
      return { failed, forgotten: yield* removals.state }
    }))
    expect(failed?._tag).toBe("RemoveFailed")
    expect(forgotten.size).toBe(0)
  })
})
