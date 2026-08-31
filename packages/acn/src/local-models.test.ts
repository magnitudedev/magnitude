import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { ModelIdSchema } from "@magnitudedev/acn-protocol"
import type {
  CatalogInstallationOperation,
  CatalogModel,
  ModelAssessmentsSnapshot,
  ReadyModel,
} from "@magnitudedev/icn-protocol/schemas"
import {
  catalogAcquisition,
  catalogModelServingState,
  catalogRemovalAcquisition,
  coordinatedAssessment,
} from "./local-models"

describe("local model serving projection", () => {
  it("accepts capabilities already decoded by the ICN client", () => {
    const ready = {
      profile: { contextLength: 32_768 },
      metadata: {
        format: "gguf",
        architecture: "test",
        quantization: "q4_k_m",
        quantizationName: "Q4_K_M",
        storageBytes: 1,
        maximumContextLength: Option.some(32_768),
      },
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: true,
        reasoning: {
          supported: true,
          efforts: ["high"],
          defaultEffort: Option.some("high"),
        },
      },
      speculativeMethod: Option.none(),
    } as unknown as ReadyModel
    const assessment = {
      _tag: "Assessed",
      assessment: { _tag: "Fits" },
    } as unknown as Parameters<typeof catalogModelServingState>[1]

    const state = catalogModelServingState(ready, assessment, Option.none())

    expect(state._tag).toBe("Assessed")
    if (state._tag !== "Assessed") return
    expect(state.capabilities.reasoning.defaultEffort).toEqual(Option.some("high"))
  })

  it("preserves ICN's native unavailability failure", () => {
    const nativeFailure = {
      code: "invalid_artifact",
      message: "The selected GGUF artifact is invalid",
      retryable: false,
    }
    expect(catalogModelServingState(
      undefined,
      undefined,
      Option.none(),
      nativeFailure,
    )).toEqual({
      _tag: "Failed",
      profile: Option.none(),
      failure: nativeFailure,
    })
  })
})

describe("automatic assessment projection", () => {
  const modelId = ModelIdSchema.make("test-model:gguf:q4")
  const failure = { code: "temporary", message: "temporary failure", retryable: true }

  it("keeps preparation, pending work, and stale source revisions assessing", () => {
    expect(coordinatedAssessment(
      { revision: 0, state: { _tag: "Preparing" } } as ModelAssessmentsSnapshot,
      1,
      "catalog",
      modelId,
    )).toBeUndefined()
    const pending = {
      revision: 1,
      state: {
        _tag: "Ready",
        environmentId: "environment",
        catalog: { _tag: "Pending", sourceRevision: 1 },
        discovered: { _tag: "Pending", sourceRevision: 1 },
      },
    } as ModelAssessmentsSnapshot
    expect(coordinatedAssessment(pending, 1, "catalog", modelId)).toBeUndefined()
    expect(coordinatedAssessment(pending, 2, "catalog", modelId)).toBeUndefined()
  })

  it("projects pool, source, and exact-target failures", () => {
    const poolFailure = {
      revision: 1,
      state: { _tag: "Failed", failure },
    } as ModelAssessmentsSnapshot
    expect(coordinatedAssessment(poolFailure, 1, "catalog", modelId)).toEqual({
      _tag: "Failed",
      failure,
    })

    const sourceFailure = {
      revision: 2,
      state: {
        _tag: "Ready",
        environmentId: "environment",
        catalog: { _tag: "Failed", sourceRevision: 1, failure },
        discovered: { _tag: "Pending", sourceRevision: 1 },
      },
    } as ModelAssessmentsSnapshot
    expect(coordinatedAssessment(sourceFailure, 1, "catalog", modelId)).toEqual({
      _tag: "Failed",
      failure,
    })

    const targetFailure = {
      revision: 3,
      state: {
        _tag: "Ready",
        environmentId: "environment",
        catalog: {
          _tag: "Complete",
          sourceRevision: 1,
          totalTargets: 1,
          failedTargets: 1,
          entries: [{
            subject: { _tag: "Catalog", modelId, selection: "Desired" },
            state: { _tag: "Failed", failure },
          }],
        },
        discovered: { _tag: "Pending", sourceRevision: 1 },
      },
    } as ModelAssessmentsSnapshot
    expect(coordinatedAssessment(targetFailure, 1, "catalog", modelId)).toEqual({
      _tag: "Failed",
      failure,
    })
  })
})

describe("catalog removal projection", () => {
  const installed = {
    _tag: "Installed" as const,
    installation: {
      _tag: "Resolved" as const,
      installedBytes: 1,
      primaryPath: "/models/model.gguf",
      ownership: "Magnitude" as const,
    },
    residencyState: { _tag: "Unloaded" as const },
  }

  it("projects admitted and failed removals without losing installed evidence", () => {
    expect(catalogRemovalAcquisition(installed, { _tag: "Removing" })).toMatchObject({
      _tag: "Removing",
      installation: installed.installation,
    })
    const failure = { code: "remove_failed", message: "Removal failed", retryable: true }
    expect(catalogRemovalAcquisition(installed, { _tag: "RemoveFailed", failure })).toMatchObject({
      _tag: "RemoveFailed",
      installation: installed.installation,
      failure,
    })
  })

  it("does not let a rejected removal hide an active update", () => {
    const updating = {
      _tag: "Updating" as const,
      installation: installed.installation,
      residencyState: installed.residencyState,
      progress: {
        stage: "downloading" as const,
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.some(1),
      },
    }
    expect(catalogRemovalAcquisition(updating, { _tag: "Removing" })).toBe(updating)
    expect(catalogRemovalAcquisition(updating, {
      _tag: "RemoveFailed",
      failure: { code: "catalog_installation_active", message: "active", retryable: false },
    })).toBe(updating)
  })
})

describe("catalog acquisition projection", () => {
  it("does not let an obsolete failed occurrence override a now-current installation", () => {
    const model = {
      localState: {
        _tag: "Installed",
        installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/model.gguf", ownership: "Magnitude" },
        updateState: { _tag: "Current" },
      },
    } as unknown as CatalogModel
    const operation = {
      state: {
        _tag: "Failed",
        acknowledged: false,
        failure: { _tag: "NetworkUnavailable" },
      },
    } as unknown as CatalogInstallationOperation

    expect(catalogAcquisition(model, operation, { _tag: "Unloaded" })).toMatchObject({ _tag: "Installed" })
  })
})
