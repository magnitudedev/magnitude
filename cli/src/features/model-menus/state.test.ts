import { describe, expect, test } from "vitest"
import { Option } from "effect"
import { makeInstalledCatalogModel, makeModel, makeView } from "../local-inference/test-fixtures"
import { modelMenusLocalModelsStateEquivalent } from "./state"

const withHeadroom = (
  allocationHeadroomBytes: number,
  insufficient: boolean,
) => {
  const model = makeModel()
  if (model.state.servingState._tag !== "Assessed"
    || model.state.servingState.assessment._tag !== "Fits") {
    throw new Error("fixture must be an assessed fitting model")
  }
  return makeModel({
    state: {
      ...model.state,
      servingState: {
        ...model.state.servingState,
        assessment: {
          ...model.state.servingState.assessment,
          memory: {
            ...model.state.servingState.assessment.memory,
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
    const left = makeView({ models: [makeInstalledCatalogModel()] }).models
    const right = {
      ...left,
      models: left.models.map((model) => model._tag === "Catalog" ? ({
        ...model,
        acquisitionState: {
          _tag: "Installing" as const,
          progress: {
            stage: "downloading" as const,
            completedBytes: 1,
            totalBytes: 2,
            bytesPerSecond: Option.none(),
          },
        },
      }) : model),
    }

    expect(modelMenusLocalModelsStateEquivalent(left, right)).toBe(false)
  })
})
