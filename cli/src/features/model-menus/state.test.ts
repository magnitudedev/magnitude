import { describe, expect, test } from "vitest"
import { Option } from "effect"
import { DownloadAttemptIdSchema } from "@magnitudedev/sdk"
import { makeModel, makeView } from "../local-inference/test-fixtures"
import { modelMenusLocalModelsStateEquivalent } from "./state"

const withHeadroom = (
  allocationHeadroomBytes: number,
  insufficient: boolean,
) => {
  const model = makeModel()
  if (model.servingState._tag !== "Assessed"
    || model.servingState.assessment._tag !== "Fits") {
    throw new Error("fixture must be an assessed fitting model")
  }
  return makeModel({
    servingState: {
      ...model.servingState,
      assessment: {
        ...model.servingState.assessment,
        memory: {
          ...model.servingState.assessment.memory,
          currentHeadroomState: insufficient
            ? {
                _tag: "Insufficient",
                observation: {
                  requiredSystemMemoryBytes: 20,
                  allocationHeadroomBytes,
                  abortReserveBytes: 2,
                  loadBoundaryBytes: 22,
                },
                minimumAdditionalAvailableBytes: Math.max(1, 23 - allocationHeadroomBytes),
              }
            : {
                _tag: "Sufficient",
                observation: {
                  requiredSystemMemoryBytes: 20,
                  allocationHeadroomBytes,
                  abortReserveBytes: 2,
                  loadBoundaryBytes: 22,
                },
              },
        },
      },
    },
  })
}

describe("model-menu state equality", () => {
  test("ignores byte-only headroom polling changes", () => {
    const left = makeView({ models: [withHeadroom(10, true)] }).models
    const right = makeView({ models: [withHeadroom(12, true)] }).models

    expect(modelMenusLocalModelsStateEquivalent(left, right)).toBe(true)
  })

  test("observes a headroom category transition", () => {
    const left = makeView({ models: [withHeadroom(10, true)] }).models
    const right = makeView({ models: [withHeadroom(30, false)] }).models

    expect(modelMenusLocalModelsStateEquivalent(left, right)).toBe(false)
  })

  test("observes acquisition progress", () => {
    const left = makeView().models
    const right = {
      ...left,
      models: left.models.map((model) => ({
        ...model,
        acquisitionState: {
          _tag: "Downloading" as const,
          attemptIds: [DownloadAttemptIdSchema.make("attempt")] as const,
          stage: "downloading" as const,
          completedBytes: 1,
          totalBytes: 2,
          bytesPerSecond: Option.none(),
        },
      })),
    }

    expect(modelMenusLocalModelsStateEquivalent(left, right)).toBe(false)
  })
})
