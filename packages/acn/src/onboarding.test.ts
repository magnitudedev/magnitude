import { Effect } from "effect"
import { describe, expect, it } from "vitest"
import type { LocalModel } from "@magnitudedev/acn-protocol"
import { resolveOnboardingState } from "./onboarding"

const modelWithOwnership = (ownership: "Magnitude" | "External" | "Mixed"): LocalModel => ({
  _tag: "Catalog",
  modelId: "lfm3-1.7b:gguf:q8",
  acquisitionState: {
    _tag: "Installed",
    installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/models/lfm3.gguf", ownership },
    residencyState: { _tag: "Unloaded" },
  },
} as unknown as LocalModel)

const foundExternalModel = (): LocalModel => ({
  _tag: "Custom",
  modelId: "local:lmstudio",
  origin: { _tag: "Found", source: { _tag: "LmStudio" } },
  acquisitionState: {
    _tag: "External",
    installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/lmstudio/model.gguf", ownership: "External" },
    residencyState: { _tag: "Unloaded" },
  },
} as unknown as LocalModel)

const notInstalledModel = (): LocalModel => ({
  _tag: "Catalog",
  modelId: "lfm3-1.7b:gguf:q8",
  acquisitionState: { _tag: "NotInstalled" },
} as unknown as LocalModel)

describe("resolveOnboardingState", () => {
  it("auto-completes onboarding on upgrade when saved chats already exist", async () => {
    let completed = 0
    const result = await Effect.runPromise(resolveOnboardingState({
      readSaved: Effect.succeed({ onboarding: "Pending" }),
      markCompleted: Effect.sync(() => { completed++ }),
      hasExistingChats: Effect.succeed(true),
      localModels: Effect.succeed([]),
    }))
    expect(result).toEqual({ onboarding: "Completed" })
    expect(completed).toBe(1)
  })

  it("auto-completes onboarding on upgrade when a Magnitude-downloaded model already exists", async () => {
    for (const ownership of ["Magnitude", "Mixed"] as const) {
      let completed = 0
      const result = await Effect.runPromise(resolveOnboardingState({
        readSaved: Effect.succeed({ onboarding: "Pending" }),
        markCompleted: Effect.sync(() => { completed++ }),
        hasExistingChats: Effect.succeed(false),
        localModels: Effect.succeed([modelWithOwnership(ownership)]),
      }))
      expect(result).toEqual({ onboarding: "Completed" })
      expect(completed).toBe(1)
    }
  })

  it("keeps onboarding pending on a fresh install with 0 Magnitude downloads, even when LM Studio / external models exist", async () => {
    let completed = 0
    const result = await Effect.runPromise(resolveOnboardingState({
      readSaved: Effect.succeed({ onboarding: "Pending" }),
      markCompleted: Effect.sync(() => { completed++ }),
      hasExistingChats: Effect.succeed(false),
      localModels: Effect.succeed([notInstalledModel(), foundExternalModel(), modelWithOwnership("External")]),
    }))
    expect(result).toEqual({ onboarding: "Pending" })
    expect(completed).toBe(0)
  })

  it("returns Completed immediately without querying chats or models once saved as Completed", async () => {
    let queried = false
    const result = await Effect.runPromise(resolveOnboardingState({
      readSaved: Effect.succeed({ onboarding: "Completed" }),
      markCompleted: Effect.sync(() => { queried = true }),
      hasExistingChats: Effect.sync(() => { queried = true; return false }),
      localModels: Effect.sync(() => { queried = true; return [] }),
    }))
    expect(result).toEqual({ onboarding: "Completed" })
    expect(queried).toBe(false)
  })
})
