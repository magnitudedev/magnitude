import { describe, expect, it } from "vitest"
import { Cause, Effect, Option, Stream } from "effect"
import { Result } from "@effect-atom/atom-react"
import {
  DownloadAttemptIdSchema,
  ModelInstanceIdSchema,
  ModelOfferingTargetIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type LocalModelDownload,
  type LocalModelsState,
  type ModelSlotsState,
} from "@magnitudedev/sdk"
import {
  OnboardingIdle,
  OnboardingModelMachine,
  initialObservationCorrelation,
  observeAdmittedDownload,
  reduceDownloadObservation,
  reduceLoadObservation,
  requestOnboardingCancellation,
} from "./onboarding-model-machine"

const targetId = ModelOfferingTargetIdSchema.make("target_test")
const admittedAttempt = DownloadAttemptIdSchema.make("attempt_admitted")
const replacementAttempt = DownloadAttemptIdSchema.make("attempt_replacement")
const instanceId = ModelInstanceIdSchema.make("instance_admitted")
const replacementInstanceId = ModelInstanceIdSchema.make("instance_replacement")
const providerModelId = ProviderModelIdSchema.make("model_test")
const submission = {
  _tag: "DownloadThenLoad" as const,
  choice: {
    targetId,
    providerModelId,
    displayName: "Test model",
    reasoningEffort: ReasoningEffortSchema.make("none"),
  },
}

const modelsState = (download: LocalModelDownload): LocalModelsState => ({
  models: [{ targetId, download }],
} as unknown as LocalModelsState)

const slotsState = (
  id: typeof instanceId,
  lifecycle: "Loading" | "Ready" | "Stopped",
): ModelSlotsState => ({
  slots: {
    primary: {
      _tag: "ConfiguredLocal",
      selection: { providerId: "local", providerModelId },
      instance: Option.some({
        id,
        lifecycle: lifecycle === "Loading"
          ? { _tag: "Loading" }
          : lifecycle === "Ready"
            ? { _tag: "Ready" }
            : { _tag: "Stopped" },
      }),
    },
  },
} as unknown as ModelSlotsState)

describe("OnboardingModelMachine", () => {
  it("represents cancellation request, acceptance, and failure as distinct transitions", () => {
    const admitting = OnboardingModelMachine.transition(new OnboardingIdle(), "AdmittingDownload", {
      submission,
      cancellationRequested: false,
    })
    const admitted = OnboardingModelMachine.transition(admitting, "DownloadAdmitted", {
      attemptIds: [admittedAttempt],
    })
    const requesting = OnboardingModelMachine.transition(
      admitted,
      "RequestingDownloadCancellation",
      {},
    )

    expect(OnboardingModelMachine.transition(
      requesting,
      "AwaitingDownloadCancellation",
      {},
    )._tag).toBe("AwaitingDownloadCancellation")
    expect(OnboardingModelMachine.transition(
      requesting,
      "DownloadCancellationFailed",
      {},
    )._tag).toBe("DownloadCancellationFailed")
  })

  it("does not permit cancellation failure to transition directly to successful cancellation", () => {
    const admitting = OnboardingModelMachine.transition(new OnboardingIdle(), "Assigning", {
      submission,
      cancellationRequested: false,
    })
    const loading = OnboardingModelMachine.transition(admitting, "AdmittingLoad", {
      cancellationRequested: false,
    })
    const admitted = OnboardingModelMachine.transition(loading, "LoadAdmitted", { instanceId })
    const requesting = OnboardingModelMachine.transition(admitted, "RequestingLoadStop", {})
    const failed = OnboardingModelMachine.transition(requesting, "LoadStopFailed", {})

    expect(() => {
      // @ts-expect-error This deliberately verifies the runtime guard as well as the type-level guard.
      OnboardingModelMachine.transition(failed, "AwaitingLoadStop", {})
    }).toThrow("Invalid FSM transition")
  })

  it("retains pre-admission cancellation as intent and retries only from a failure state", () => {
    const admitting = OnboardingModelMachine.transition(new OnboardingIdle(), "AdmittingDownload", {
      submission,
      cancellationRequested: false,
    })
    const deferred = requestOnboardingCancellation(admitting)
    expect(deferred).toMatchObject({
      _tag: "Deferred",
      state: { _tag: "AdmittingDownload", cancellationRequested: true },
    })

    const admitted = OnboardingModelMachine.transition(admitting, "DownloadAdmitted", {
      attemptIds: [admittedAttempt],
    })
    const requesting = requestOnboardingCancellation(admitted)
    expect(requesting._tag).toBe("Download")
    if (requesting._tag !== "Download") throw new Error("Expected a download cancellation")
    const failed = OnboardingModelMachine.transition(
      requesting.state,
      "DownloadCancellationFailed",
      {},
    )
    expect(requestOnboardingCancellation(failed)._tag).toBe("Download")
  })
})

describe("admitted model observation", () => {
  it("ignores stale download identities until the exact admission is observed", () => {
    const [staleCorrelation, stale] = reduceDownloadObservation(
      initialObservationCorrelation,
      modelsState({
        _tag: "Failed",
        attemptIds: [replacementAttempt],
        completedBytes: 0,
        totalBytes: 1,
        failure: { code: "stale", message: "stale", retryable: true },
      }),
      targetId,
      [admittedAttempt],
    )
    expect(Option.isNone(stale)).toBe(true)

    const [seenCorrelation, active] = reduceDownloadObservation(
      staleCorrelation,
      modelsState({
        _tag: "Downloading",
        attemptIds: [admittedAttempt],
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.none(),
      }),
      targetId,
      [admittedAttempt],
    )
    expect(seenCorrelation.exactIdentitySeen).toBe(true)
    expect(Option.isNone(active)).toBe(true)

    const [, superseded] = reduceDownloadObservation(
      seenCorrelation,
      modelsState({
        _tag: "Downloading",
        attemptIds: [replacementAttempt],
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.none(),
      }),
      targetId,
      [admittedAttempt],
    )
    expect(Option.getOrNull(superseded)).toBe("Superseded")
  })

  it("accepts the desired downloaded condition without requiring attempt history", () => {
    const [, observation] = reduceDownloadObservation(
      initialObservationCorrelation,
      modelsState({ _tag: "Downloaded", installedBytes: 2 }),
      targetId,
      [admittedAttempt],
    )
    expect(Option.getOrNull(observation)).toBe("Downloaded")
  })

  it("does not treat an invalidated query's stale successful value as post-admission truth", async () => {
    const stale = Result.waiting(Result.success({
      state: modelsState({ _tag: "Downloaded", installedBytes: 2 }),
    }))
    const active = Result.success({
      state: modelsState({
        _tag: "Downloading",
        attemptIds: [admittedAttempt],
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.none(),
      }),
    })
    const failed = Result.success({
      state: modelsState({
        _tag: "Failed",
        attemptIds: [admittedAttempt],
        completedBytes: 1,
        totalBytes: 2,
        failure: { code: "failed", message: "failed", retryable: true },
      }),
    })

    const observation = await Effect.runPromise(observeAdmittedDownload(
      Stream.fromIterable([
        stale,
        Result.failure<{ readonly state: LocalModelsState }, string>(
          Cause.fail("query unavailable"),
        ),
        active,
        failed,
      ]),
      targetId,
      [admittedAttempt],
    ))
    expect(observation).toBe("Failed")
  })

  it("ignores stale instances and detects replacement only after exact correlation", () => {
    const [staleCorrelation, stale] = reduceLoadObservation(
      initialObservationCorrelation,
      slotsState(replacementInstanceId, "Ready"),
      submission,
      instanceId,
    )
    expect(Option.isNone(stale)).toBe(true)

    const [seenCorrelation, loading] = reduceLoadObservation(
      staleCorrelation,
      slotsState(instanceId, "Loading"),
      submission,
      instanceId,
    )
    expect(seenCorrelation.exactIdentitySeen).toBe(true)
    expect(Option.isNone(loading)).toBe(true)

    const [, superseded] = reduceLoadObservation(
      seenCorrelation,
      slotsState(replacementInstanceId, "Ready"),
      submission,
      instanceId,
    )
    expect(Option.getOrNull(superseded)).toBe("Superseded")
  })
})
