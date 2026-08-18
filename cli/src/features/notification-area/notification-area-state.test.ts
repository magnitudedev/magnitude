import { describe, expect, test } from "vitest"
import { Option } from "effect"
import { ModelDownloadIdSchema } from "@magnitudedev/sdk"
import {
  deriveModelDownloadNotificationState,
  deriveSelectedModelLowMemoryNotificationState,
} from "@magnitudedev/client-common"
import { makeAcquiringModel, makeModel, makeView } from "../local-inference/test-fixtures"

describe("notification area model projections", () => {
  test("derives download activity directly from local model state", () => {
    const view = makeView({
      models: [makeAcquiringModel({
        _tag: "Downloading",
        downloadId: ModelDownloadIdSchema.make("download-1"),
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.none(),
      })],
    })

    expect(deriveModelDownloadNotificationState(view.models)?.message)
      .toBe("1 model downloading")
  })

  test("derives low memory for the selected model only while insufficient", () => {
    const ordinaryModel = makeModel()
    if (ordinaryModel.servingState._tag !== "Assessed"
      || ordinaryModel.servingState.assessment._tag !== "Fits") {
      throw new Error("fixture must be an assessed fitting model")
    }
    const lowMemoryModel = makeModel({
      servingState: {
        ...ordinaryModel.servingState,
        assessment: {
          ...ordinaryModel.servingState.assessment,
          memory: {
            ...ordinaryModel.servingState.assessment.memory,
            currentHeadroomState: {
              _tag: "Insufficient",
              observation: {
                requiredSystemMemoryBytes: 20,
                allocationHeadroomBytes: 10,
                abortReserveBytes: 2,
                loadBoundaryBytes: 22,
              },
              minimumAdditionalAvailableBytes: 2 * 1024 ** 3 + 1,
            },
          },
        },
      },
    })
    const lowMemoryView = makeView({ models: [lowMemoryModel] })

    expect(deriveSelectedModelLowMemoryNotificationState(
      lowMemoryView.models,
      lowMemoryView.slots,
    )).toMatchObject({
      message: "Low memory: close memory-intensive apps (need 2.1 GB) to load model",
      compactMessage: Option.some("Low memory: Free 2.1 GB to load"),
      priority: "warning",
    })

    const ordinaryView = makeView({ models: [ordinaryModel] })
    expect(deriveSelectedModelLowMemoryNotificationState(
      ordinaryView.models,
      ordinaryView.slots,
    )).toBeNull()
  })
})
