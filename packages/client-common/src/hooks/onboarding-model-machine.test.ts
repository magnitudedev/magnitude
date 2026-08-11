import { describe, expect, it } from "vitest"
import { Cause, Effect, Option, Stream } from "effect"
import { Result } from "@effect-atom/atom-react"
import {
  DownloadAttemptIdSchema,
  ModelPackageIdSchema,
  ModelInstanceIdSchema,
  ModelServingConfigurationIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type LocalModelDownloadState,
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

const admittedAttempt = DownloadAttemptIdSchema.make("attempt_admitted")
const replacementAttempt = DownloadAttemptIdSchema.make("attempt_replacement")
const instanceId = ModelInstanceIdSchema.make("instance_admitted")
const replacementInstanceId = ModelInstanceIdSchema.make("instance_replacement")
const providerModelId = ProviderModelIdSchema.make("model_test")
const configurationId = ModelServingConfigurationIdSchema.make("configuration_test")
const submission = {
  _tag: "ConfigureThenLoad" as const,
  choice: {
    configurationId,
    displayName: "Test model",
    reasoningEffort: ReasoningEffortSchema.make("none"),
  },
}

const modelsState = (download: LocalModelDownloadState): LocalModelsState => ({
  inventory: { _tag: "Ready" },
  models: [],
  downloads: [{
    configuration: {
      id: configurationId,
      bundle: {
        _tag: "Standalone",
        package: {
          id: ModelPackageIdSchema.make("package_test"),
          source: { _tag: "Local", path: "/models/test.gguf" },
          files: [],
          relationships: [],
          properties: {
            format: "gguf",
            quantization: "Q4_K_M",
            quantizationName: "4-bit",
            architecture: "test",
            maximumContextLength: 32_768,
          },
        },
      },
      profile: { contextLength: 32_768 },
    },
    presentation: { displayName: "Test model", description: "" },
    capabilities: Option.some({
      vision: false,
      tools: true,
      structuredOutput: true,
      reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
    }),
    state: download,
  }],
  recommendations: { _tag: "Loading", progress: [] },
})

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
      configurationId,
      providerModelId,
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
      providerModelId,
      cancellationRequested: false,
    })
    const loading = OnboardingModelMachine.transition(admitting, "AdmittingLoad", {
      providerModelId,
      cancellationRequested: false,
    })
    const admitted = OnboardingModelMachine.transition(loading, "LoadAdmitted", {
      providerModelId,
      instanceId,
    })
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
      configurationId,
      providerModelId,
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
      configurationId,
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
      configurationId,
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
      configurationId,
      [admittedAttempt],
    )
    expect(Option.getOrNull(superseded)).toBe("Superseded")
  })

  it("accepts the desired downloaded condition without requiring attempt history", () => {
    const [, observation] = reduceDownloadObservation(
      initialObservationCorrelation,
      modelsState({ _tag: "Downloaded", installedBytes: 2, origins: ["Magnitude"] }),
      configurationId,
      [admittedAttempt],
    )
    expect(Option.getOrNull(observation)).toBe("Downloaded")
  })

  it("does not treat an invalidated query's stale successful value as post-admission truth", async () => {
    const stale = Result.waiting(Result.success({
      state: modelsState({ _tag: "Downloaded", installedBytes: 2, origins: ["Magnitude"] }),
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
      configurationId,
      [admittedAttempt],
    ))
    expect(observation).toBe("Failed")
  })

  it("ignores stale instances and detects replacement only after exact correlation", () => {
    const [staleCorrelation, stale] = reduceLoadObservation(
      initialObservationCorrelation,
      slotsState(replacementInstanceId, "Ready"),
      providerModelId,
      instanceId,
    )
    expect(Option.isNone(stale)).toBe(true)

    const [seenCorrelation, loading] = reduceLoadObservation(
      staleCorrelation,
      slotsState(instanceId, "Loading"),
      providerModelId,
      instanceId,
    )
    expect(seenCorrelation.exactIdentitySeen).toBe(true)
    expect(Option.isNone(loading)).toBe(true)

    const [, superseded] = reduceLoadObservation(
      seenCorrelation,
      slotsState(replacementInstanceId, "Ready"),
      providerModelId,
      instanceId,
    )
    expect(Option.getOrNull(superseded)).toBe("Superseded")
  })
})
