import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Cause, Effect, Layer, Option } from "effect"
import { describe, expect, it } from "vitest"
import { Client as EffectQueryClient } from "@magnitudedev/effect-query"
import {
  AssessmentEnvironmentIdSchema,
  CatalogModelIdSchema,
  CatalogVariantIdSchema,
  ModelDownloadIdSchema,
  ModelAssessmentIdSchema,
  ModelInstanceIdSchema,
  ModelPackageIdSchema,
  ModelReleaseDateSchema,
  ModelServingConfigurationIdSchema,
  ModelSlotConfiguredLocal,
  ModelSlotUnassigned,
  ModelVariantLabelSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  RecommendationIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  type LocalModel,
  type ModelDownloadFailure,
  type LocalModelsState,
  type ModelSlotsState,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { clientServicesLayer, type ClientServices } from "../state/client-services"
import type { AcnClientRequirements } from "../state/agent-client"
import { fakeAcnImplementationsLayer } from "../state/fake-acn-implementations"
import { localModelProviderModelId } from "./projection"
import { installationAdmissionIsVisible } from "./service"
import { OnboardingModelSetup } from "./setup"
import { localModelOptions } from "./options"
import {
  projectOnboardingModelSetupContent,
  type OnboardingModelSetupState,
} from "./setup-state"

const configurationId = ModelServingConfigurationIdSchema.make("setup-configuration")
const providerModelId = ProviderModelIdSchema.make("setup-model:gguf:q4")
const instanceId = ModelInstanceIdSchema.make("setup-instance")
const downloadId = ModelDownloadIdSchema.make("setup-attempt")
const reasoningEffort = ReasoningEffortSchema.make("none")
const localProviderId = ProviderIdSchema.make("local")
const allocation = {
  contextWindowTokens: 32_768,
  parallelSequences: 1,
  physicalContextTokens: 32_768,
  memoryDomains: [],
}

const makeModel = (installed: boolean): LocalModel => {
  const bundle = {
    _tag: "Standalone" as const,
    package: {
      id: ModelPackageIdSchema.make("setup-package"),
      source: { _tag: "Local" as const, path: "/models/setup.gguf" },
      files: [],
      relationships: [],
      properties: {
        format: "gguf",
        quantization: "Q4_K_M",
        quantizationName: "4-bit",
        architecture: "test",
        maximumContextLength: Option.some(32_768),
        intrinsicModelId: Option.none(),
        intrinsicQualityId: Option.none(),
      },
    },
  }
  return {
    bundle,
    presentation: {
      displayName: "Setup Model",
      variantLabel: ModelVariantLabelSchema.make("Q4"),
      description: "",
      license: Option.none(),
    },
    downloadBytes: 1,
    catalogMembershipState: {
      _tag: "InCatalog",
      catalogData: {
        modelId: CatalogModelIdSchema.make("setup-model"),
        variantId: CatalogVariantIdSchema.make("gguf:q4"),
        releaseDate: ModelReleaseDateSchema.make("2026-01-01"),
        parameterization: { architecture: "dense", totalParameters: 1 },
        intelligenceScore: 1,
        intelligenceScoreSource: "test",
        fidelityRank: 1,
        quantizationAware: false,
        qualityNotes: [],
      },
    },
    upgradeState: installed ? { _tag: "Current" } : { _tag: "NotApplicable" },
    acquisitionState: installed
      ? {
          _tag: "Installed",
          installedBytes: 1,
          packages: [{ packageId: bundle.package.id, path: "/models/setup.gguf", origin: "Magnitude" }],
        }
      : { _tag: "NotInstalled", completedBytes: 0, totalBytes: 1 },
    servingState: {
      _tag: "Assessed",
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: true,
        reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
      },
      configuration: { id: configurationId, bundle, profile: { contextLength: 32_768 } },
      assessment: {
        _tag: "Fits",
        profile: { contextLength: 32_768 },
        assessmentId: ModelAssessmentIdSchema.make("setup-assessment"),
        environmentId: AssessmentEnvironmentIdSchema.make("setup-environment"),
        memory: {
          domains: [],
          totalRequiredBytes: 0,
          requiredSystemMemoryBytes: 0,
          systemUseState: {
            _tag: "WithinRecommendedHeadroom",
            recommendedHeadroomBytes: 0,
            predictedHeadroomBytes: 0,
          },
          currentHeadroomState: { _tag: "NotObserved" },
        },
        performance: [],
      },
      availabilityState: installed
        ? { _tag: "Selectable", providerModelId }
        : { _tag: "Installable" },
      recommendations: installed ? [] : [{
        id: RecommendationIdSchema.make("setup-recommendation"),
        intent: "balanced",
        explanation: "Recommended for setup tests",
      }],
    },
  }
}

const selection: SlotSelection = {
  providerId: localProviderId,
  providerModelId,
  reasoningEffort,
}

const unassignedSlots = (): ModelSlotsState => ({
  slots: {
    primary: new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
    secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
  },
  recentModels: { primary: [], secondary: [] },
  favoriteModels: [],
})

const configuredSlots = (
  lifecycle: "None" | "Loading" | "Ready" | "Stopped",
  id = instanceId,
  activeConfigurationId = configurationId,
): ModelSlotsState => ({
  ...unassignedSlots(),
  slots: {
    ...unassignedSlots().slots,
    primary: new ModelSlotConfiguredLocal({
      slotId: PRIMARY_SLOT_ID,
      selection,
      descriptor: {
        providerId: localProviderId,
        providerModelId,
        displayName: "Setup Model",
        variantLabel: Option.some(ModelVariantLabelSchema.make("Q4")),
      },
      availability: { _tag: "Available" },
      residency: lifecycle === "None" || lifecycle === "Stopped"
        ? { _tag: "Unloaded" }
        : lifecycle === "Ready"
          ? { _tag: "Ready", instanceId: id, configurationId: activeConfigurationId, allocation }
          : {
              _tag: "Loading",
              instanceId: id,
              configurationId: activeConfigurationId,
              stage: "loading",
              progress: Option.none(),
              plannedAllocation: Option.none(),
            },
      actions: lifecycle === "None" || lifecycle === "Stopped" ? ["Load"] : ["Stop"],
    }),
  },
})

describe("projectOnboardingModelSetupContent", () => {
  it("does not report a different serving configuration as ready", () => {
    const model = makeModel(true)
    const option = localModelOptions({
      inventoryState: { _tag: "Ready" },
      models: [model],
      discoveryState: { _tag: "Ready", progress: [] },
    }, unassignedSlots())[0]!
    const state = projectOnboardingModelSetupContent(
      Option.some({
        _tag: "Loading",
        option,
        configurationId,
        providerModelId,
        selection,
        cancelling: false,
      }),
      {
        inventoryState: { _tag: "Ready" },
        models: [model],
        discoveryState: { _tag: "Ready", progress: [] },
      },
      configuredSlots(
        "Ready",
        instanceId,
        ModelServingConfigurationIdSchema.make("replacement-configuration"),
      ),
    )

    expect(Option.getOrThrow(state._tag === "Chooser" ? state.operation : Option.none()))
      .toMatchObject({ _tag: "Loading", phase: "Loading" })
  })

  it("does not let discovery refresh mask an active invocation", () => {
    const model = makeModel(true)
    const option = localModelOptions({
      inventoryState: { _tag: "Ready" },
      models: [model],
      discoveryState: { _tag: "Ready", progress: [] },
    }, configuredSlots("Loading"))[0]!
    const state = projectOnboardingModelSetupContent(
      Option.some({
        _tag: "Loading",
        option,
        configurationId,
        providerModelId,
        selection,
        cancelling: false,
      }),
      {
        inventoryState: { _tag: "Ready" },
        models: [model],
        discoveryState: { _tag: "Loading", progress: [] },
      },
      configuredSlots("Loading"),
    )

    expect(Option.getOrThrow(state._tag === "Chooser" ? state.operation : Option.none()))
      .toMatchObject({ _tag: "Loading" })
  })

  it("retains the submitted model when discovery temporarily removes it", () => {
    const model = makeModel(true)
    const option = localModelOptions({
      inventoryState: { _tag: "Ready" },
      models: [model],
      discoveryState: { _tag: "Ready", progress: [] },
    }, configuredSlots("Loading"))[0]!
    const state = projectOnboardingModelSetupContent(
      Option.some({
        _tag: "Loading",
        option,
        configurationId,
        providerModelId,
        selection,
        cancelling: false,
      }),
      {
        inventoryState: { _tag: "Ready" },
        models: [],
        discoveryState: { _tag: "Loading", progress: [] },
      },
      configuredSlots("Loading"),
    )

    expect(Option.getOrThrow(state._tag === "Chooser" ? state.operation : Option.none()))
      .toMatchObject({ _tag: "Loading", model: { presentation: { displayName: "Setup Model" } } })
  })
})

describe("installationAdmissionIsVisible", () => {
  it("accepts the exact admitted model download before a provider identity exists", () => {
    const uninstalled = makeModel(false)
    const downloading: LocalModel = {
      ...uninstalled,
      acquisitionState: {
        _tag: "Downloading",
        downloadId,
        stage: "downloading",
        completedBytes: 0,
        totalBytes: 1,
        bytesPerSecond: Option.none(),
      },
    }
    const state: LocalModelsState = {
      inventoryState: { _tag: "Ready" },
      models: [downloading],
      discoveryState: { _tag: "Ready", progress: [] },
    }

    expect(installationAdmissionIsVisible(state, configurationId, {
      _tag: "DownloadAdmitted",
      providerModelId,
      downloadId,
    })).toBe(true)
    expect(Option.isNone(localModelProviderModelId(downloading))).toBe(true)
  })

  it("rejects a different model-download occurrence for the same configuration", () => {
    const uninstalled = makeModel(false)
    const state: LocalModelsState = {
      inventoryState: { _tag: "Ready" },
      models: [{
        ...uninstalled,
        acquisitionState: {
          _tag: "Downloading",
          downloadId: ModelDownloadIdSchema.make("replacement-download"),
          stage: "downloading",
          completedBytes: 0,
          totalBytes: 1,
          bytesPerSecond: Option.none(),
        },
      }],
      discoveryState: { _tag: "Ready", progress: [] },
    }

    expect(installationAdmissionIsVisible(state, configurationId, {
      _tag: "DownloadAdmitted",
      providerModelId,
      downloadId,
    })).toBe(false)
  })

  it("accepts a current installed configuration while provider publication is pending", () => {
    const installed = makeModel(true)
    const publishing = installed.servingState._tag === "Assessed" ? {
      ...installed,
      servingState: {
        ...installed.servingState,
        availabilityState: { _tag: "Preparing" as const, providerModelId },
      },
    } : installed

    expect(installationAdmissionIsVisible({
      inventoryState: { _tag: "Ready" },
      models: [publishing],
      discoveryState: { _tag: "Ready", progress: [] },
    }, configurationId, {
      _tag: "Current",
      providerModelId,
    })).toBe(true)
  })

  it("does not complete update admission until the exact occurrence is visible", () => {
    const installed = {
      ...makeModel(true),
      upgradeState: {
        _tag: "Available" as const,
        missingPackageIds: [],
        supersededPackageIds: [],
      },
    }
    const state = {
      inventoryState: { _tag: "Ready" as const },
      models: [installed],
      discoveryState: { _tag: "Ready" as const, progress: [] },
    }
    const admission = { _tag: "DownloadAdmitted" as const, providerModelId, downloadId }
    expect(installationAdmissionIsVisible(state, configurationId, admission)).toBe(false)
    expect(installationAdmissionIsVisible({
      ...state,
      models: [{
        ...installed,
        upgradeState: {
          _tag: "Upgrading",
          downloadId,
          stage: "downloading",
          completedBytes: 0,
          totalBytes: 1,
          bytesPerSecond: Option.none(),
        },
      }],
    }, configurationId, admission)).toBe(true)
  })
})

interface HarnessOptions {
  readonly installed: boolean
  readonly onboardingCompleted?: boolean
  readonly initiallyOpen?: boolean
  readonly initiallyDownloading?: boolean
  readonly ready?: boolean
  readonly keepLoading?: boolean
  readonly keepLoadRequest?: boolean
  readonly keepDownloading?: boolean
  readonly failAssign?: boolean
  readonly keepCompleting?: boolean
  readonly failCompletion?: boolean
  readonly failInitialOnboardingRead?: boolean
  readonly failInitialLocalModelsRead?: boolean
  readonly failInitialModelSlotsRead?: boolean
  readonly replaceSelectionBeforeLoad?: boolean
  readonly downloadFailure?: ModelDownloadFailure
}

const makeHarness = (options: HarnessOptions) => {
  let model = makeModel(options.installed)
  if (options.initiallyDownloading && model.servingState._tag === "Assessed") {
    model = {
      ...model,
      acquisitionState: {
        _tag: "Downloading",
        downloadId,
        stage: "downloading",
        completedBytes: 0,
        totalBytes: 1,
        bytesPerSecond: Option.none(),
      },
      servingState: {
        ...model.servingState,
        availabilityState: { _tag: "Installable" },
      },
    }
  }
  let models: LocalModelsState = {
    inventoryState: { _tag: "Ready" },
    models: [model],
    discoveryState: { _tag: "Ready", progress: [] },
  }
  let slots = options.ready ? configuredSlots("Ready") : unassignedSlots()
  let onboardingCompleted = options.onboardingCompleted ?? false
  let revision = 0
  const calls: string[] = []
  const stoppedInstances: unknown[] = []
  const cancelledDownloads: unknown[] = []
  const rpc = ((name: string, payload: any) => Effect.suspend<unknown, unknown, never>(() => {
    calls.push(name)
    switch (name) {
      case "GetLocalModels": {
        if (options.failInitialLocalModelsRead
          && calls.filter((call) => call === "GetLocalModels").length === 1) {
          return Effect.fail({ _tag: "LocalModelsQueryFailed", message: "temporarily unavailable" })
        }
        return Effect.succeed({ revision: revision++, state: models })
      }
      case "GetModelSlots": {
        if (options.failInitialModelSlotsRead
          && calls.filter((call) => call === "GetModelSlots").length === 1) {
          return Effect.fail({ _tag: "ModelSlotsQueryFailed", message: "temporarily unavailable" })
        }
        return Effect.succeed({ revision: revision++, state: slots })
      }
      case "GetOnboardingState": {
        if (options.failInitialOnboardingRead
          && calls.filter((call) => call === "GetOnboardingState").length === 1) {
          return Effect.fail({
            _tag: "OnboardingError",
            operation: "get onboarding state",
            message: "temporarily unavailable",
          })
        }
        return Effect.succeed({
          revision: revision++,
          state: { completed: onboardingCompleted },
        })
      }
      case "ReconcileCatalogModel": {
        model = options.downloadFailure !== undefined
          ? (() => {
              const uninstalled = makeModel(false)
              return {
                ...uninstalled,
                acquisitionState: {
                  _tag: "Failed" as const,
                  downloadId,
                  completedBytes: 0,
                  totalBytes: 1,
                  failure: options.downloadFailure,
                },
              }
            })()
          : options.keepDownloading
          ? (() => {
              const uninstalled = makeModel(false)
              return uninstalled.servingState._tag === "Assessed" ? {
                ...uninstalled,
                acquisitionState: {
                  _tag: "Downloading" as const,
                  downloadId,
                  stage: "downloading" as const,
                  completedBytes: 0,
                  totalBytes: 1,
                  bytesPerSecond: Option.none(),
                },
                servingState: {
                  ...uninstalled.servingState,
                  availabilityState: { _tag: "Installable" as const },
                },
              } : uninstalled
            })()
          : makeModel(true)
        models = { ...models, models: [model] }
        return Effect.succeed({
          _tag: "DownloadAdmitted",
          providerModelId,
          downloadId,
        })
      }
      case "AssignSlot": {
        if (options.failAssign) return Effect.fail({
          _tag: "ModelSlotMutationRejected" as const,
          slotId: PRIMARY_SLOT_ID,
          message: "assignment rejected",
        })
        slots = configuredSlots("None")
        return Effect.succeed({})
      }
      case "LoadModel":
        if (options.replaceSelectionBeforeLoad) {
          slots = configuredSlots("None")
          return Effect.fail({
            _tag: "ModelSlotMutationRejected" as const,
            slotId: PRIMARY_SLOT_ID,
            message: "The selected local model changed before load admission",
          })
        }
        slots = configuredSlots(
          options.keepLoading ? "Loading" : "Ready",
          instanceId,
        )
        return options.keepLoadRequest ? Effect.never : Effect.succeed({})
      case "StopModel":
        stoppedInstances.push(instanceId)
        slots = configuredSlots("Stopped", instanceId)
        return Effect.succeed({})
      case "CancelModelDownload":
        cancelledDownloads.push(payload.downloadId)
        models = {
          ...models,
          models: [{
            ...model,
            acquisitionState: {
              _tag: "Cancelled",
              downloadId,
              completedBytes: 0,
              totalBytes: 1,
            },
          }],
        }
        return Effect.succeed({})
      case "CompleteOnboarding":
        if (options.keepCompleting) return Effect.never
        if (options.failCompletion) return Effect.fail({
          _tag: "OnboardingError",
          operation: "complete onboarding",
          message: "completion failed",
        })
        onboardingCompleted = true
        return Effect.succeed({})
      default: return Effect.die(new Error(`Unexpected RPC ${name}`))
    }
  }))
  const effectQuery = EffectQueryClient.make<AcnClientRequirements, never, ClientServices, never>(
    fakeAcnImplementationsLayer(rpc),
    (client) => clientServicesLayer(client, {
      onboardingSetupInitiallyOpen: options.initiallyOpen,
    }),
  )
  const registry = Registry.make()
  const serviceReference = effectQuery.runtime.atom(OnboardingModelSetup)
  const service = {
    view: Atom.make((get) => Result.flatMap(
      get(serviceReference),
      (setup) => get(setup.view),
    )),
    retry: effectQuery.runtime.fn(
      () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.retry),
      { concurrent: true },
    ),
    open: effectQuery.runtime.fn(
      () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.open),
      { concurrent: true },
    ),
    select: effectQuery.runtime.fn<typeof configurationId>()(
      (input) => Effect.flatMap(OnboardingModelSetup, (setup) => setup.select(input)),
      { concurrent: true },
    ),
    cancel: effectQuery.runtime.fn(
      () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.cancel),
      { concurrent: true },
    ),
    exit: effectQuery.runtime.fn(
      () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.exit),
      { concurrent: true },
    ),
  }
  return {
    calls,
    registry,
    service,
    stoppedInstances,
    cancelledDownloads,
    onboardingCompleted: () => onboardingCompleted,
  }
}

const execute = <Input, Output, Error>(
  registry: Registry.Registry,
  command: Atom.Writable<Result.Result<Output, Error>, Input | Atom.Reset | Atom.Interrupt>,
  input: Input,
) => Effect.async<Output, Error>((resume) => {
  let settled = false
  const unsubscribe = registry.subscribe(command, (result) => {
    if (settled || result.waiting || Result.isInitial(result)) return
    settled = true
    unsubscribe()
    resume(Result.isSuccess(result)
      ? Effect.succeed(result.value)
      : Effect.failCause(result.cause))
  })
  registry.set(command, input)
  return Effect.sync(unsubscribe)
})

const waitForCall = (calls: readonly string[], name: string): Effect.Effect<void> =>
  calls.includes(name)
    ? Effect.void
    : Effect.sleep("1 millis").pipe(Effect.zipRight(Effect.suspend(() => waitForCall(calls, name))))

const waitForView = (
  harness: ReturnType<typeof makeHarness>,
  predicate: (state: OnboardingModelSetupState) => boolean,
  attempts = 1_000,
): Effect.Effect<OnboardingModelSetupState> => Effect.suspend(() => {
  const state = Result.value(harness.registry.get(harness.service.view))
  return Option.isSome(state) && predicate(state.value)
    ? Effect.succeed(state.value)
    : attempts <= 0
      ? Effect.die(new Error(`View did not converge: ${JSON.stringify({
          calls: harness.calls,
          state: Option.getOrNull(state),
        })}`))
      : Effect.sleep("1 millis").pipe(
          Effect.zipRight(waitForView(harness, predicate, attempts - 1)),
        )
})

describe("OnboardingModelSetup", () => {
  it("retries the onboarding observation through its own query", async () => {
    const harness = makeHarness({
      installed: true,
      ready: true,
      failInitialOnboardingRead: true,
    })
    const unmount = harness.registry.mount(harness.service.view)
    await Effect.runPromise(waitForCall(harness.calls, "GetOnboardingState"))
    const readsBeforeRetry = harness.calls.filter((call) => call === "GetOnboardingState").length

    await Effect.runPromise(execute(harness.registry, harness.service.retry, undefined))
    await Effect.runPromise(Effect.sleep("5 millis"))

    expect(harness.calls.filter((call) => call === "GetOnboardingState")).toHaveLength(
      readsBeforeRetry + 1,
    )
    unmount()
    harness.registry.dispose()
  })

  it.each([
    ["local-model", "GetLocalModels", { failInitialLocalModelsRead: true }],
    ["model-slot", "GetModelSlots", { failInitialModelSlotsRead: true }],
  ] as const)("retries only the failed %s observation", async (_, failedCall, failureOption) => {
    const harness = makeHarness({ installed: true, ready: true, ...failureOption })
    const unmount = harness.registry.mount(harness.service.view)
    await Effect.runPromise(waitForCall(harness.calls, failedCall))
    await Effect.runPromise(Effect.sleep("5 millis"))
    const beforeRetry = {
      onboarding: harness.calls.filter((call) => call === "GetOnboardingState").length,
      models: harness.calls.filter((call) => call === "GetLocalModels").length,
      slots: harness.calls.filter((call) => call === "GetModelSlots").length,
    }

    await Effect.runPromise(execute(harness.registry, harness.service.retry, undefined))
    await Effect.runPromise(Effect.sleep("5 millis"))

    expect(harness.calls.filter((call) => call === failedCall)).toHaveLength(
      (failedCall === "GetLocalModels" ? beforeRetry.models : beforeRetry.slots) + 1,
    )
    expect(harness.calls.filter((call) => call === "GetOnboardingState")).toHaveLength(
      beforeRetry.onboarding,
    )
    const unaffectedCall = failedCall === "GetLocalModels" ? "GetModelSlots" : "GetLocalModels"
    expect(harness.calls.filter((call) => call === unaffectedCall)).toHaveLength(
      unaffectedCall === "GetLocalModels" ? beforeRetry.models : beforeRetry.slots,
    )
    unmount()
    harness.registry.dispose()
  })

  it("is observational and passive across state remounts", async () => {
    const harness = makeHarness({ installed: true, ready: true })
    const firstUnmount = harness.registry.mount(harness.service.view)
    await Effect.runPromise(Effect.sleep("5 millis"))
    firstUnmount()
    const secondUnmount = harness.registry.mount(harness.service.view)
    await Effect.runPromise(Effect.sleep("5 millis"))
    const state = Result.value(harness.registry.get(harness.service.view))
    expect(Option.isSome(state) && state.value._tag).toBe("Open")
    if (Option.isSome(state) && state.value._tag === "Open") {
      expect(state.value.content._tag).toBe("Chooser")
    }
    secondUnmount()

    expect(harness.calls.filter((call) => [
      "ReconcileCatalogModel",
      "AssignSlot",
      "LoadModel",
      "StopModel",
      "CancelModelDownload",
      "CompleteOnboarding",
    ].includes(call))).toEqual([])
    harness.registry.dispose()
  })

  it("only completes onboarding for an exact already-ready choice", async () => {
    const harness = makeHarness({ installed: true, ready: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      configurationId,
    ))
    await Effect.runPromise(waitForCall(harness.calls, "CompleteOnboarding"))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))

    expect(harness.onboardingCompleted()).toBe(true)
    expect(harness.calls).not.toContain("ReconcileCatalogModel")
    expect(harness.calls).not.toContain("AssignSlot")
    expect(harness.calls).not.toContain("LoadModel")
    harness.registry.dispose()
  })

  it("reopens completed onboarding at a fresh requested choice without completing it again", async () => {
    const harness = makeHarness({ installed: true, ready: true })
    await Effect.runPromise(execute(harness.registry, harness.service.select, configurationId))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))
    const completionCalls = () => harness.calls.filter((call) => call === "CompleteOnboarding").length

    expect(Option.getOrThrow(Result.value(harness.registry.get(harness.service.view))))
      .toEqual({ _tag: "Closed" })
    expect(Option.getOrThrow(Result.value(harness.registry.get(harness.service.view))))
      .toEqual({ _tag: "Closed" })
    expect(completionCalls()).toBe(1)

    await Effect.runPromise(execute(harness.registry, harness.service.open, undefined))
    expect(Option.getOrThrow(Result.value(harness.registry.get(harness.service.view))))
      .toMatchObject({
        _tag: "Open",
        exitKind: "Close",
        content: { _tag: "Chooser" },
      })
    expect(completionCalls()).toBe(1)

    harness.registry.set(harness.service.select, Atom.Reset)
    await Effect.runPromise(execute(harness.registry, harness.service.select, configurationId))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))
    expect(Option.getOrThrow(Result.value(harness.registry.get(harness.service.view))))
      .toEqual({ _tag: "Closed" })
    expect(completionCalls()).toBe(1)
    harness.registry.dispose()
  })

  it("applies launch-open intent before publishing the first resolved view", async () => {
    const harness = makeHarness({ installed: true, onboardingCompleted: true, initiallyOpen: true })
    const resolvedTags: OnboardingModelSetupState["_tag"][] = []
    const unsubscribe = harness.registry.subscribe(harness.service.view, (result) => {
      if (Result.isSuccess(result)) resolvedTags.push(result.value._tag)
    }, { immediate: true })

    const state = await Effect.runPromise(waitForView(
      harness,
      (view) => view._tag === "Open",
    ))

    expect(state).toMatchObject({
      _tag: "Open",
      exitKind: "Close",
      content: { _tag: "Chooser" },
    })
    expect(resolvedTags).not.toContain("Closed")
    expect(harness.calls).not.toContain("CompleteOnboarding")
    unsubscribe()
    harness.registry.dispose()
  })

  it("never publishes an open view for completed onboarding without launch intent", async () => {
    const harness = makeHarness({ installed: true, onboardingCompleted: true })
    const resolvedTags: OnboardingModelSetupState["_tag"][] = []
    const unsubscribe = harness.registry.subscribe(harness.service.view, (result) => {
      if (Result.isSuccess(result)) resolvedTags.push(result.value._tag)
    }, { immediate: true })

    await Effect.runPromise(waitForView(harness, (view) => view._tag === "Closed"))

    expect(resolvedTags).not.toContain("Open")
    unsubscribe()
    harness.registry.dispose()
  })

  it("does not start model work after onboarding is closed until it is reopened", async () => {
    const harness = makeHarness({ installed: true, ready: true, onboardingCompleted: true })
    const start = await Effect.runPromiseExit(execute(
      harness.registry,
      harness.service.select,
      configurationId,
    ))

    expect(start._tag).toBe("Failure")
    if (start._tag === "Failure") {
      expect(Cause.pretty(start.cause)).toContain("OnboardingModelSetupNotOpen")
    }
    expect(harness.calls.filter((call) => [
      "ReconcileCatalogModel",
      "AssignSlot",
      "LoadModel",
      "CompleteOnboarding",
    ].includes(call))).toEqual([])
    harness.registry.dispose()
  })

  it("closes requested setup without changing durable onboarding", async () => {
    const harness = makeHarness({ installed: true, ready: true, onboardingCompleted: true })
    await Effect.runPromise(execute(harness.registry, harness.service.open, undefined))
    await Effect.runPromise(execute(harness.registry, harness.service.exit, undefined))

    expect(harness.onboardingCompleted()).toBe(true)
    expect(harness.calls).not.toContain("CompleteOnboarding")
    expect(Option.getOrThrow(Result.value(harness.registry.get(harness.service.view))))
      .toEqual({ _tag: "Closed" })
    harness.registry.dispose()
  })

  it("retains a required-exit failure in visible onboarding state", async () => {
    const harness = makeHarness({ installed: true, failCompletion: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.exit,
      undefined,
    ))

    const state = await Effect.runPromise(waitForView(
      harness,
      (view) => view._tag === "Open"
        && view.content._tag === "Chooser"
        && Option.isSome(view.notice),
    ))
    expect(state).toMatchObject({
      _tag: "Open",
      exitKind: "Skip",
      content: { _tag: "Chooser" },
    })
    if (state._tag === "Open" && state.content._tag === "Chooser") {
      expect(Option.getOrThrow(state.notice)).toMatchObject({
        _tag: "OnboardingError",
        message: "completion failed",
      })
    }
    harness.registry.dispose()
  })

  it("does not erase a retained exit failure when setup is opened again", async () => {
    const harness = makeHarness({ installed: true, failCompletion: true })
    await Effect.runPromise(execute(harness.registry, harness.service.exit, undefined))
    const before = await Effect.runPromise(waitForView(
      harness,
      (view) => view._tag === "Open"
        && view.content._tag === "Chooser"
        && Option.isSome(view.notice),
    ))

    await Effect.runPromise(execute(harness.registry, harness.service.open, undefined))

    expect(Option.getOrThrow(Result.value(harness.registry.get(harness.service.view))))
      .toEqual(before)
    harness.registry.dispose()
  })

  it("projects required exit without observing model discovery or slots", async () => {
    const harness = makeHarness({ installed: false, keepCompleting: true })
    await Effect.runPromise(execute(harness.registry, harness.service.exit, undefined))
    await Effect.runPromise(waitForCall(harness.calls, "CompleteOnboarding"))

    expect(Option.getOrThrow(Result.value(harness.registry.get(harness.service.view))))
      .toEqual({
        _tag: "Open",
        exitKind: "Skip",
        notice: Option.none(),
        content: { _tag: "Closing" },
      })
    expect(harness.calls).not.toContain("GetLocalModels")
    expect(harness.calls).not.toContain("GetModelSlots")

    harness.registry.dispose()
  })

  it("keeps dormant completed setup independent of model queries", async () => {
    const harness = makeHarness({ installed: true, ready: true, onboardingCompleted: true })
    const unmount = harness.registry.mount(harness.service.view)
    await Effect.runPromise(Effect.sleep("5 millis"))

    expect(Option.getOrThrow(Result.value(harness.registry.get(harness.service.view))))
      .toEqual({ _tag: "Closed" })
    expect(harness.calls).not.toContain("GetLocalModels")
    expect(harness.calls).not.toContain("GetModelSlots")
    unmount()
    harness.registry.dispose()
  })

  it("does not admit cancellation once an already-ready choice is completing", async () => {
    const harness = makeHarness({ installed: true, ready: true, keepCompleting: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      configurationId,
    ))
    await Effect.runPromise(waitForCall(harness.calls, "CompleteOnboarding"))

    const cancellation = await Effect.runPromiseExit(execute(
      harness.registry,
      harness.service.cancel,
      undefined,
    ))
    expect(cancellation._tag).toBe("Failure")
    if (cancellation._tag === "Failure") {
      expect(Cause.pretty(cancellation.cause)).toContain("OnboardingModelSetupCancellationUnavailable")
    }
    harness.registry.dispose()
  })

  it("assigns, loads, awaits, and completes an installed non-ready choice", async () => {
    const harness = makeHarness({ installed: true })
    await Effect.runPromise(execute(harness.registry, harness.service.select, configurationId))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))

    const mutations = harness.calls.filter((call) => [
      "ReconcileCatalogModel",
      "AssignSlot",
      "LoadModel",
      "CompleteOnboarding",
    ].includes(call))
    expect(mutations).toEqual(["AssignSlot", "LoadModel", "CompleteOnboarding"])
    harness.registry.dispose()
  })

  it("installs before assignment for an uninstalled choice", async () => {
    const harness = makeHarness({ installed: false })
    await Effect.runPromise(execute(harness.registry, harness.service.select, configurationId))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))

    const mutations = harness.calls.filter((call) => [
      "ReconcileCatalogModel",
      "AssignSlot",
      "LoadModel",
      "CompleteOnboarding",
    ].includes(call))
    expect(mutations).toEqual([
      "ReconcileCatalogModel",
      "AssignSlot",
      "LoadModel",
      "CompleteOnboarding",
    ])
    harness.registry.dispose()
  })

  it("preserves a structured download failure as the terminal setup failure", async () => {
    const failure: ModelDownloadFailure = {
      _tag: "InsufficientDiskSpace",
      requiredBytes: 40,
      availableBytes: 30,
    }
    const harness = makeHarness({ installed: false, downloadFailure: failure })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      configurationId,
    ))

    const state = await Effect.runPromise(waitForView(
      harness,
      (view) => view._tag === "Open"
        && view.content._tag === "Chooser"
        && Option.isSome(view.notice),
    ))
    if (state._tag !== "Open" || state.content._tag !== "Chooser") {
      throw new Error("expected open chooser")
    }
    expect(Option.getOrThrow(state.notice)).toEqual(failure)
    expect(harness.calls).not.toContain("AssignSlot")
    expect(harness.calls).not.toContain("LoadModel")
    harness.registry.dispose()
  })

  it("does not mistake a pre-existing download provider identity for installation", async () => {
    const harness = makeHarness({ installed: false, initiallyDownloading: true })
    await Effect.runPromise(execute(harness.registry, harness.service.select, configurationId))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))

    expect(harness.calls.indexOf("ReconcileCatalogModel")).toBeGreaterThanOrEqual(0)
    expect(harness.calls.indexOf("ReconcileCatalogModel")).toBeLessThan(harness.calls.indexOf("AssignSlot"))
    harness.registry.dispose()
  })

  it("stops after a failed dependency and never runs downstream mutations", async () => {
    const harness = makeHarness({ installed: true, failAssign: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      configurationId,
    ))

    await Effect.runPromise(waitForView(
      harness,
      (view) => view._tag === "Open"
        && view.content._tag === "Chooser"
        && Option.isSome(view.notice),
    ))
    expect(harness.calls).toContain("AssignSlot")
    expect(harness.calls).not.toContain("LoadModel")
    expect(harness.calls).not.toContain("CompleteOnboarding")
    harness.registry.dispose()
  })

  it("does not load through a slot selection replacement", async () => {
    const harness = makeHarness({ installed: true, replaceSelectionBeforeLoad: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      configurationId,
    ))

    await Effect.runPromise(waitForView(
      harness,
      (view) => view._tag === "Open"
        && view.content._tag === "Chooser"
        && Option.isSome(view.notice),
    ))
    expect(harness.calls).toContain("LoadModel")
    expect(harness.calls).not.toContain("CompleteOnboarding")
    expect(harness.onboardingCompleted()).toBe(false)
    harness.registry.dispose()
  })

  it("cancels only the exact admitted download and waits for its terminal fact", async () => {
    const harness = makeHarness({ installed: false, keepDownloading: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      configurationId,
    ))
    await Effect.runPromise(waitForCall(harness.calls, "ReconcileCatalogModel"))
    await Effect.runPromise(execute(harness.registry, harness.service.cancel, undefined))

    expect(harness.cancelledDownloads).toEqual([downloadId])
    expect(harness.calls).not.toContain("AssignSlot")
    expect(harness.calls).not.toContain("LoadModel")
    harness.registry.dispose()
  })

  it("sends stop while the load request is still pending", async () => {
    const harness = makeHarness({ installed: true, keepLoading: true, keepLoadRequest: true })
    await Effect.runPromise(execute(harness.registry, harness.service.select, configurationId))
    await Effect.runPromise(waitForCall(harness.calls, "LoadModel"))

    await Effect.runPromise(execute(harness.registry, harness.service.cancel, undefined))

    expect(harness.calls.indexOf("StopModel")).toBeGreaterThan(harness.calls.indexOf("LoadModel"))
    expect(harness.stoppedInstances).toEqual([instanceId])
    expect(harness.onboardingCompleted()).toBe(false)
    harness.registry.dispose()
  })

  it("exiting required setup completes onboarding without touching any model resource", async () => {
    const harness = makeHarness({ installed: false })
    await Effect.runPromise(execute(harness.registry, harness.service.exit, undefined))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))

    expect(harness.onboardingCompleted()).toBe(true)
    expect(harness.calls.filter((call) => [
      "ReconcileCatalogModel",
      "AssignSlot",
      "LoadModel",
      "StopModel",
      "CancelModelDownload",
    ].includes(call))).toEqual([])
    harness.registry.dispose()
  })

  it("keeps active setup alive when its React-style observer unmounts", async () => {
    const harness = makeHarness({ installed: true, keepLoading: true })
    const unmount = harness.registry.mount(harness.service.view)
    await Effect.runPromise(execute(harness.registry, harness.service.select, configurationId))
    await Effect.runPromise(waitForCall(harness.calls, "LoadModel"))
    unmount()
    await Effect.runPromise(Effect.sleep("5 millis"))

    expect(harness.calls).not.toContain("CompleteOnboarding")
    await Effect.runPromise(execute(harness.registry, harness.service.cancel, undefined))
    expect(harness.stoppedInstances).toEqual([instanceId])
    harness.registry.dispose()
  })

})
