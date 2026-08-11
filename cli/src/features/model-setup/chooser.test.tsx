import { describe, expect, it } from "vitest"
import { Option } from "effect"
import { DownloadAttemptIdSchema } from "@magnitudedev/sdk"
import { makeAcquiringModel } from "../local-inference/test-fixtures"
import {
  scrollOnboardingModelIntoView,
  type OnboardingModelChooserOperation,
} from "./chooser"

describe("onboarding model chooser identity", () => {
  it("scrolls by presentation identity without copying model fields", () => {
    const calls: string[] = []
    scrollOnboardingModelIntoView({
      scrollChildIntoView: (id: string) => { calls.push(id) },
    } as never, "model-id")
    expect(calls).toEqual(["onboarding-model:model-id"])
  })

  it("does nothing before the model viewport is mounted", () => {
    expect(() => scrollOnboardingModelIntoView(null, "model-id")).not.toThrow()
  })

  it("carries the canonical model through acquisition operations", () => {
    const model = makeAcquiringModel({
      _tag: "Downloading",
      attemptIds: [DownloadAttemptIdSchema.make("attempt")],
      stage: "downloading",
      completedBytes: 1,
      totalBytes: 2,
      bytesPerSecond: Option.none(),
    })
    const operation: OnboardingModelChooserOperation = {
      _tag: "Downloading",
      model,
      cancelling: false,
      cancelError: null,
      onCancel: () => undefined,
      onRetry: () => undefined,
    }
    expect(operation.model).toBe(model)
  })
})
