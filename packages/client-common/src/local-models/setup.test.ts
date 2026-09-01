import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Cause, Deferred, Effect, Layer, Option, Queue, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { Client as EffectQueryClient } from "@magnitudedev/effect-query"
import {
  MagnitudeBoundary,
  AssessmentEnvironmentIdSchema,
  CatalogIntelligenceSchema,
  CatalogFormModelIdSchema,
  ModelAssessmentIdSchema,
  ModelReleaseDateSchema,
  ModelSlotConfiguredLocal,
  ModelSlotUnassigned,
  ModelVariantLabelSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  type LocalModel,
  type ModelAcquisitionFailure,
  type ModelFailure,
  type ModelInstanceFailure,
  type LocalModelsState,
  type ModelSlotsState,
  type SlotSelection,
  type Change,
} from "@magnitudedev/sdk"
import { clientServicesLayer, type ClientServices } from "../state/client-services"
import type { AcnClientRequirements } from "../state/agent-client"
import { fakeAcnImplementationsLayer } from "../state/fake-acn-implementations"
import { localModelProviderModelId } from "./projection"
import { OnboardingModelSetup } from "./setup"
import { localModelOptions } from "./options"
import { onboardingModelSetupNoticeMessage } from "./failure-messages"
import {
  defaultOnboardingModelRankingControls,
  normalizeOnboardingModelRankingControls,
  projectOnboardingModelSetupContent,
  type OnboardingModelSetupState,
} from "./setup-state"

const providerModelId = CatalogFormModelIdSchema.make("setup-model:gguf:q4")
const instanceId = "setup-instance"
const reasoningEffort = ReasoningEffortSchema.make("none")
const localProviderId = ProviderIdSchema.make("local")
const allocation = {
  contextWindowTokens: 32_768,
  parallelSequences: 1,
  physicalContextTokens: 32_768,
  memoryDomains: [],
}
const completePreparation = {
  discovery: { complete: true, modelsFound: 1 },
  assessment: { complete: true, settledModels: 1, totalModels: 1 },
} as const
const activePreparation = {
  discovery: { complete: false, modelsFound: 3 },
  assessment: { complete: false, settledModels: 2, totalModels: 4 },
} as const

const makeModel = (installed: boolean): Extract<LocalModel, { readonly _tag: "Catalog" }> => {
  return {
    _tag: "Catalog",
    modelId: providerModelId,
    storageBytes: 1,
    presentation: {
      displayName: "Setup Model",
      variantLabel: ModelVariantLabelSchema.make("Q4"),
      description: "",
      license: Option.none(),
      sourceUrls: [],
    },
    catalogData: {
        releaseDate: ModelReleaseDateSchema.make("2026-01-01"),
        parameterization: { architecture: "dense", totalParameters: 1 },
        intelligence: Schema.decodeUnknownSync(CatalogIntelligenceSchema)({
          score: 1,
          provenance: {
            kind: "artificialAnalysisIntelligenceIndex",
            methodologyVersion: "test",
            asOfDate: "2026-01-01",
            url: "https://example.com/model",
          },
        }),
        fidelityRank: 1,
        quantizationAware: false,
    },
    acquisitionState: installed
      ? {
          _tag: "Installed",
          installation: {
            _tag: "Resolved",
            installedBytes: 1,
            primaryPath: "/models/setup.gguf",
            ownership: "Magnitude",
          },
          residencyState: { _tag: "Unloaded" },
        }
      : { _tag: "NotInstalled" },
    servingState: {
      _tag: "Assessed",
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: true,
        reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
      },
      speculativeMethod: Option.none(),
      metadata: {
        format: "gguf",
        quantization: "Q4_K_M",
        quantizationName: "4-bit",
        architecture: "test",
        maximumContextLength: Option.some(32_768),
        storageBytes: 1,
      },
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
      rankingScores: installed
        ? Option.none()
        : Option.some({ intelligence: 0.7, speed: 0.8, fidelity: 0.9 }),
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
  lifecycle: "None" | "Loading" | "Ready" | "Stopped" | "Failed",
  id = instanceId,
  loadingProgress: Option.Option<number> = Option.none(),
  failure?: ModelInstanceFailure,
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
        : lifecycle === "Failed"
          ? { _tag: "Failed", failure: failure ?? {
              code: "load_failed",
              message: "load failed",
              retryable: true,
            } }
        : lifecycle === "Ready"
          ? { _tag: "Ready", allocation }
          : {
              _tag: "Loading",
              stage: "loading",
              progress: loadingProgress,
              plannedAllocation: Option.none(),
            },
      actions: lifecycle === "None" || lifecycle === "Stopped" ? ["Load"] : ["Stop"],
    }),
  },
})

describe("projectOnboardingModelSetupContent", () => {
  it("clamps the connection-scoped ranking preference", () => {
    expect(normalizeOnboardingModelRankingControls({
      fastToSmart: 2,
    })).toEqual({ fastToSmart: 1 })
  })

  it("keeps the exact live discovery and assessment counts in preparation", () => {
    const content = projectOnboardingModelSetupContent(
      Option.none(),
      {
        preparation: activePreparation,
        models: [makeModel(true)],
      },
      unassignedSlots(),
      defaultOnboardingModelRankingControls,
    )

    expect(content).toEqual({ _tag: "Preparation", preparation: activePreparation })
  })

  it("waits for assessment after discovery completes and reveals the chooser only when both complete", () => {
    const model = makeModel(true)
    const assessing = projectOnboardingModelSetupContent(
      Option.none(),
      {
        preparation: {
          discovery: { complete: true, modelsFound: 4 },
          assessment: { complete: false, settledModels: 3, totalModels: 4 },
        },
        models: [model],
      },
      unassignedSlots(),
      defaultOnboardingModelRankingControls,
    )
    const ready = projectOnboardingModelSetupContent(
      Option.none(),
      { preparation: completePreparation, models: [model] },
      unassignedSlots(),
      defaultOnboardingModelRankingControls,
    )

    expect(assessing._tag).toBe("Preparation")
    expect(ready._tag).toBe("Chooser")
  })

  it("reports the selected model's ready instance", () => {
    const model = makeModel(true)
    const option = localModelOptions({
      preparation: completePreparation,
      models: [model],
    }, unassignedSlots())[0]!
    const state = projectOnboardingModelSetupContent(
      Option.some({
        _tag: "Loading",
        option,
        modelId: providerModelId,
        providerModelId,
        selection,
        cancelling: false,
      }),
      {
        preparation: completePreparation,
        models: [model],
      },
      configuredSlots("Ready", instanceId),
      defaultOnboardingModelRankingControls,
    )

    expect(Option.getOrThrow(state._tag === "Chooser" ? state.operation : Option.none()))
      .toMatchObject({ _tag: "Loading", status: { _tag: "Ready" } })
  })

  it("preserves authoritative model loading progress", () => {
    const model = makeModel(true)
    const slots = configuredSlots("Loading", instanceId, Option.some(0.42))
    const option = localModelOptions({
      preparation: completePreparation,
      models: [model],
    }, slots)[0]!
    const state = projectOnboardingModelSetupContent(
      Option.some({
        _tag: "Loading",
        option,
        modelId: providerModelId,
        providerModelId,
        selection,
        cancelling: false,
      }),
      {
        preparation: completePreparation,
        models: [model],
      },
      slots,
      defaultOnboardingModelRankingControls,
    )

    expect(Option.getOrThrow(state._tag === "Chooser" ? state.operation : Option.none()))
      .toMatchObject({
        _tag: "Loading",
        status: { _tag: "Loading", stage: "loading", progress: Option.some(0.42) },
      })
  })

  it("distinguishes cancellation from ordinary model stopping", () => {
    const model = makeModel(true)
    const slots = configuredSlots("Loading", instanceId, Option.some(0.42))
    const option = localModelOptions({
      preparation: completePreparation,
      models: [model],
    }, slots)[0]!
    const state = projectOnboardingModelSetupContent(
      Option.some({
        _tag: "Loading",
        option,
        modelId: providerModelId,
        providerModelId,
        selection,
        cancelling: true,
      }),
      {
        preparation: completePreparation,
        models: [model],
      },
      slots,
      defaultOnboardingModelRankingControls,
    )

    expect(Option.getOrThrow(state._tag === "Chooser" ? state.operation : Option.none()))
      .toMatchObject({ _tag: "Loading", status: { _tag: "Cancelling" } })
  })

  it("does not let discovery refresh mask an active invocation", () => {
    const model = makeModel(true)
    const option = localModelOptions({
      preparation: completePreparation,
      models: [model],
    }, configuredSlots("Loading"))[0]!
    const state = projectOnboardingModelSetupContent(
      Option.some({
        _tag: "Loading",
        option,
        modelId: providerModelId,
        providerModelId,
        selection,
        cancelling: false,
      }),
      {
        models: [model],
        preparation: activePreparation,
      },
      configuredSlots("Loading"),
      defaultOnboardingModelRankingControls,
    )

    expect(Option.getOrThrow(state._tag === "Chooser" ? state.operation : Option.none()))
      .toMatchObject({ _tag: "Loading" })
  })

  it("retains the submitted model when discovery temporarily removes it", () => {
    const model = makeModel(true)
    const option = localModelOptions({
      preparation: completePreparation,
      models: [model],
    }, configuredSlots("Loading"))[0]!
    const state = projectOnboardingModelSetupContent(
      Option.some({
        _tag: "Loading",
        option,
        modelId: providerModelId,
        providerModelId,
        selection,
        cancelling: false,
      }),
      {
        models: [],
        preparation: activePreparation,
      },
      configuredSlots("Loading"),
      defaultOnboardingModelRankingControls,
    )

    expect(Option.getOrThrow(state._tag === "Chooser" ? state.operation : Option.none()))
      .toMatchObject({ _tag: "Loading", model: { presentation: { displayName: "Setup Model" } } })
  })
})

interface HarnessOptions {
  readonly installed: boolean
  readonly onboardingCompleted?: boolean
  readonly initiallyOpen?: boolean
  readonly initiallyDownloading?: boolean
  readonly ready?: boolean
  readonly keepLoading?: boolean
  readonly keepDownloading?: boolean
  readonly failAssign?: boolean
  readonly keepCompleting?: boolean
  readonly failCompletion?: boolean
  readonly failInitialOnboardingRead?: boolean
  readonly failInitialLocalModelsRead?: boolean
  readonly failInitialModelSlotsRead?: boolean
  readonly replaceSelectionBeforeLoad?: boolean
  readonly postSyncCatalogRead?: Deferred.Deferred<void>
  readonly downloadFailure?: ModelAcquisitionFailure
  readonly loadFailure?: ModelInstanceFailure
}

const makeHarness = (options: HarnessOptions) => {
  let model = makeModel(options.installed)
  if (options.initiallyDownloading && model.servingState._tag === "Assessed") {
    model = {
      ...model,
      acquisitionState: {
        _tag: "Installing",
        progress: {
          stage: "downloading",
          completedBytes: 0,
          totalBytes: 1,
          bytesPerSecond: Option.none(),
        },
      },
    }
  }
  let models: LocalModelsState = {
    preparation: completePreparation,
    models: [model],
  }
  let slots = options.ready ? configuredSlots("Ready") : unassignedSlots()
  let onboardingCompleted = options.onboardingCompleted ?? false
  const calls: string[] = []
  const stoppedInstances: unknown[] = []
  const cancelledDownloads: unknown[] = []
  const changes = Effect.runSync(Queue.unbounded<Change>())
  const providerCatalog = () => ({
      _tag: "Ready" as const,
      providers: [{
        providerId: localProviderId,
        displayName: "Local",
        kind: "Local" as const,
        authentication: "NotRequired" as const,
        availability: { _tag: "Available" as const },
      }],
      models: [{
        providerId: localProviderId,
        providerModelId,
        modelFamilyId: Option.none(),
        displayName: "Setup Model",
        variantLabel: Option.some(ModelVariantLabelSchema.make("Q4")),
        supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
        contextWindow: 32_768,
        maxOutputTokens: 32_768,
        memory: Option.none(),
        capabilities: model.servingState._tag === "Assessed"
          ? model.servingState.capabilities
          : {
              vision: false,
              tools: true,
              structuredOutput: true,
              reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
            },
        availability: model.acquisitionState._tag === "Installed"
          ? { _tag: "Available" as const }
          : { _tag: "Disabled" as const, reason: "installation_unavailable" as const },
        pricing: Option.none(),
      }],
  })
  const modelCatalog = () => {
    const providers = providerCatalog()
    return {
      _tag: "Ready" as const,
      providers: providers.providers,
      models: [{
        _tag: "Local" as const,
        product: model,
        offering: Option.fromNullable(providers.models[0]),
      }],
      failures: [],
      localModelPreparation: models.preparation,
    }
  }
  const rpc = ((name: string, payload: any) => Effect.suspend<unknown, unknown, never>(() => {
    calls.push(name)
    switch (name) {
      case "GetModelCatalog": {
        if (options.failInitialLocalModelsRead
          && calls.filter((call) => call === "GetModelCatalog").length === 1) {
          return Effect.fail({ _tag: "LocalModelsQueryFailed", message: "temporarily unavailable" })
        }
        return options.postSyncCatalogRead !== undefined && calls.includes("SyncLocalModel")
          ? Deferred.await(options.postSyncCatalogRead).pipe(Effect.as(modelCatalog()))
          : Effect.succeed(modelCatalog())
      }
      case "GetModelSlots": {
        if (options.failInitialModelSlotsRead
          && calls.filter((call) => call === "GetModelSlots").length === 1) {
          return Effect.fail({ _tag: "ModelSlotsQueryFailed", message: "temporarily unavailable" })
        }
        return Effect.succeed(slots)
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
        return Effect.succeed({ completed: onboardingCompleted })
      }
      case "SyncLocalModel": {
        model = options.downloadFailure !== undefined
          ? (() => {
              const uninstalled = makeModel(false)
              return {
                ...uninstalled,
                acquisitionState: {
                  _tag: "InstallFailed" as const,
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
                  _tag: "Installing" as const,
                  progress: {
                    stage: "downloading" as const,
                    completedBytes: 0,
                    totalBytes: 1,
                    bytesPerSecond: Option.none(),
                  },
                },
              } : uninstalled
            })()
          : makeModel(true)
        models = { ...models, models: [model] }
        return Queue.offer(changes, { query: "GetModelCatalog" }).pipe(
          Effect.as({ outcome: "Started" as const }),
        )
      }
      case "AssignModelSlot": {
        if (options.failAssign) return Effect.fail({
          _tag: "ModelSlotMutationRejected" as const,
          slotId: PRIMARY_SLOT_ID,
          message: "assignment rejected",
        })
        slots = configuredSlots("None")
        return Queue.offer(changes, { query: "GetModelSlots" }).pipe(Effect.as({}))
      }
      case "LoadModelSlot":
        if (options.replaceSelectionBeforeLoad) {
          slots = configuredSlots("None")
          return Effect.fail({
            _tag: "ModelSlotMutationRejected" as const,
            slotId: PRIMARY_SLOT_ID,
            message: "The selected local model changed before load admission",
          })
        }
        if (options.loadFailure !== undefined) {
          slots = configuredSlots("Failed", instanceId, Option.none(), options.loadFailure)
          return Queue.offer(changes, { query: "GetModelSlots" }).pipe(
            Effect.zipRight(Effect.fail({
              _tag: "LocalModelMutationFailed" as const,
              code: options.loadFailure.code,
              message: options.loadFailure.message,
              retryable: options.loadFailure.retryable,
            })),
          )
        }
        slots = configuredSlots(
          options.keepLoading ? "Loading" : "Ready",
          instanceId,
        )
        // The ICN ensure operation owns acquisition and does not return until
        // the instance is ready. Cancelling this client wait must only detach
        // from the request; the shared ICN acquisition continues.
        return Queue.offer(changes, { query: "GetModelSlots" }).pipe(
          Effect.zipRight(options.keepLoading ? Effect.never : Effect.succeed({})),
        )
      case "StopModelSlot":
        stoppedInstances.push(instanceId)
        slots = configuredSlots("Stopped", instanceId)
        return Queue.offer(changes, { query: "GetModelSlots" }).pipe(Effect.as({}))
      case "CancelLocalModelSync":
        cancelledDownloads.push(payload.modelId)
        models = {
          ...models,
          models: [{
            ...model,
            acquisitionState: { _tag: "NotInstalled" },
          }],
        }
        return Queue.offer(changes, { query: "GetModelCatalog" }).pipe(Effect.as({}))
      case "CompleteOnboarding":
        if (options.keepCompleting) return Effect.never
        if (options.failCompletion) return Effect.fail({
          _tag: "OnboardingError",
          operation: "complete onboarding",
          message: "completion failed",
        })
        onboardingCompleted = true
        return Queue.offer(changes, { query: "GetOnboardingState" }).pipe(Effect.as({}))
      default: return Effect.die(new Error(`Unexpected RPC ${name}`))
    }
  }))
  const effectQuery = EffectQueryClient.make<typeof MagnitudeBoundary, AcnClientRequirements, never, ClientServices, never>(
    MagnitudeBoundary,
    fakeAcnImplementationsLayer(
      rpc,
      (name) => name === "StreamChanges" ? Stream.fromQueue(changes) : Stream.never,
    ),
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
    select: effectQuery.runtime.fn<typeof providerModelId>()(
      (input) => Effect.flatMap(OnboardingModelSetup, (setup) => setup.select(input)),
      { concurrent: true },
    ),
    cancel: effectQuery.runtime.fn(
      () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.cancel),
      { concurrent: true },
    ),
    chooseAnother: effectQuery.runtime.fn(
      () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.chooseAnother),
      { concurrent: true },
    ),
    continueWithMagnitude: effectQuery.runtime.fn(
      () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.continueWithHarness(
        "magnitude" as never,
        { launchOnStartup: false, installSkill: false },
      )),
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
    ["local-model", "GetModelCatalog", { failInitialLocalModelsRead: true }],
    ["model-slot", "GetModelSlots", { failInitialModelSlotsRead: true }],
  ] as const)("retries only the failed %s observation", async (_, failedCall, failureOption) => {
    const harness = makeHarness({ installed: true, ready: true, ...failureOption })
    const unmount = harness.registry.mount(harness.service.view)
    await Effect.runPromise(waitForCall(harness.calls, failedCall))
    await Effect.runPromise(Effect.sleep("5 millis"))
    const beforeRetry = {
      onboarding: harness.calls.filter((call) => call === "GetOnboardingState").length,
      models: harness.calls.filter((call) => call === "GetModelCatalog").length,
      slots: harness.calls.filter((call) => call === "GetModelSlots").length,
    }

    await Effect.runPromise(execute(harness.registry, harness.service.retry, undefined))
    await Effect.runPromise(Effect.sleep("5 millis"))

    expect(harness.calls.filter((call) => call === failedCall)).toHaveLength(
      (failedCall === "GetModelCatalog" ? beforeRetry.models : beforeRetry.slots) + 1,
    )
    expect(harness.calls.filter((call) => call === "GetOnboardingState")).toHaveLength(
      beforeRetry.onboarding,
    )
    const unaffectedCall = failedCall === "GetModelCatalog" ? "GetModelSlots" : "GetModelCatalog"
    expect(harness.calls.filter((call) => call === unaffectedCall)).toHaveLength(
      unaffectedCall === "GetModelCatalog" ? beforeRetry.models : beforeRetry.slots,
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
      "SyncLocalModel",
      "AssignModelSlot",
      "LoadModelSlot",
      "StopModelSlot",
      "CancelLocalModelSync",
      "CompleteOnboarding",
    ].includes(call))).toEqual([])
    harness.registry.dispose()
  })

  it("only completes onboarding for an exact already-ready choice", async () => {
    const harness = makeHarness({ installed: true, ready: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      providerModelId,
    ))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Open" && state.content._tag === "Harness"))
    await Effect.runPromise(execute(harness.registry, harness.service.continueWithMagnitude, undefined))
    await Effect.runPromise(waitForCall(harness.calls, "CompleteOnboarding"))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))

    expect(harness.onboardingCompleted()).toBe(true)
    expect(harness.calls).not.toContain("SyncLocalModel")
    expect(harness.calls).not.toContain("AssignModelSlot")
    expect(harness.calls).not.toContain("LoadModelSlot")
    harness.registry.dispose()
  })

  it("does not consume the stale pre-sync catalog while the replacement read is pending", async () => {
    const postSyncCatalogRead = Effect.runSync(Deferred.make<void>())
    const harness = makeHarness({ installed: false, postSyncCatalogRead })
    const selection = Effect.runFork(execute(
      harness.registry,
      harness.service.select,
      providerModelId,
    ))

    await Effect.runPromise(waitForCall(harness.calls, "SyncLocalModel"))
    await Effect.runPromise(Effect.sleep("5 millis"))
    const pending = Option.getOrThrow(Result.value(harness.registry.get(harness.service.view)))
    expect(pending).toMatchObject({
      _tag: "Open",
      notice: { _tag: "None" },
      content: { _tag: "Chooser" },
    })

    await Effect.runPromise(Deferred.succeed(postSyncCatalogRead, undefined))
    await Effect.runPromise(selection)
    await Effect.runPromise(waitForView(
      harness,
      (state) => state._tag === "Open" && state.content._tag === "Harness",
    ))
    harness.registry.dispose()
  })

  it("reopens completed onboarding at a fresh requested choice without completing it again", async () => {
    const harness = makeHarness({ installed: true, ready: true })
    await Effect.runPromise(execute(harness.registry, harness.service.select, providerModelId))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Open" && state.content._tag === "Harness"))
    await Effect.runPromise(execute(harness.registry, harness.service.continueWithMagnitude, undefined))
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
    await Effect.runPromise(execute(harness.registry, harness.service.select, providerModelId))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Open" && state.content._tag === "Harness"))
    harness.registry.set(harness.service.continueWithMagnitude, Atom.Reset)
    await Effect.runPromise(execute(harness.registry, harness.service.continueWithMagnitude, undefined))
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
      providerModelId,
    ))

    expect(start._tag).toBe("Failure")
    if (start._tag === "Failure") {
      expect(Cause.pretty(start.cause)).toContain("OnboardingModelSetupNotOpen")
    }
    expect(harness.calls.filter((call) => [
      "SyncLocalModel",
      "AssignModelSlot",
      "LoadModelSlot",
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
        failure: {
          _tag: "OnboardingError",
          message: "completion failed",
        },
        subject: { _tag: "Setup" },
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
    expect(harness.calls).not.toContain("GetModelCatalog")
    expect(harness.calls).not.toContain("GetModelSlots")

    harness.registry.dispose()
  })

  it("keeps dormant completed setup independent of model queries", async () => {
    const harness = makeHarness({ installed: true, ready: true, onboardingCompleted: true })
    const unmount = harness.registry.mount(harness.service.view)
    await Effect.runPromise(Effect.sleep("5 millis"))

    expect(Option.getOrThrow(Result.value(harness.registry.get(harness.service.view))))
      .toEqual({ _tag: "Closed" })
    expect(harness.calls).not.toContain("GetModelCatalog")
    expect(harness.calls).not.toContain("GetModelSlots")
    unmount()
    harness.registry.dispose()
  })

  it("does not admit cancellation once an already-ready choice is completing", async () => {
    const harness = makeHarness({ installed: true, ready: true, keepCompleting: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      providerModelId,
    ))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Open" && state.content._tag === "Harness"))
    harness.registry.set(harness.service.continueWithMagnitude, undefined)
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
    await Effect.runPromise(execute(harness.registry, harness.service.select, providerModelId))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Open" && state.content._tag === "Harness"))
    await Effect.runPromise(execute(harness.registry, harness.service.continueWithMagnitude, undefined))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))

    const mutations = harness.calls.filter((call) => [
      "SyncLocalModel",
      "AssignModelSlot",
      "LoadModelSlot",
      "CompleteOnboarding",
    ].includes(call))
    expect(mutations).toEqual(["AssignModelSlot", "LoadModelSlot", "CompleteOnboarding"])
    harness.registry.dispose()
  })

  it("retains an authoritative low-memory load result for retry or choosing another model", async () => {
    const failure: ModelInstanceFailure = {
      _tag: "LowMemory",
      code: "low_memory",
      message: "not enough memory available",
      retryable: true,
      requiredSystemMemoryBytes: 8_000_000_000,
      allocationHeadroomBytes: 6_000_000_000,
      systemReserveBytes: 1_000_000_000,
      loadBoundaryBytes: 9_000_000_000,
      minimumAdditionalAvailableBytes: 3_000_000_000,
      parallelSequences: 1,
    }
    const harness = makeHarness({ installed: true, loadFailure: failure })

    await Effect.runPromise(execute(harness.registry, harness.service.select, providerModelId))
    const failed = await Effect.runPromise(waitForView(
      harness,
      (state) => state._tag === "Open"
        && state.content._tag === "Chooser"
        && Option.exists(
          state.content.operation,
          (operation) => operation._tag === "Loading" && operation.status._tag === "Failed",
        ),
    ))
    if (failed._tag !== "Open" || failed.content._tag !== "Chooser") {
      throw new Error("expected failed load result")
    }
    expect(Option.isNone(failed.notice)).toBe(true)
    expect(Option.getOrThrow(failed.content.operation)).toMatchObject({
      _tag: "Loading",
      status: { _tag: "Failed", failure },
    })

    await Effect.runPromise(execute(harness.registry, harness.service.select, providerModelId))
    await Effect.runPromise(waitForView(
      harness,
      (state) => state._tag === "Open"
        && state.content._tag === "Chooser"
        && Option.exists(
          state.content.operation,
          (operation) => operation._tag === "Loading" && operation.status._tag === "Failed",
        )
        && harness.calls.filter((call) => call === "LoadModelSlot").length === 2,
    ))

    await Effect.runPromise(execute(harness.registry, harness.service.chooseAnother, undefined))
    const choosing = await Effect.runPromise(waitForView(
      harness,
      (state) => state._tag === "Open"
        && state.content._tag === "Chooser"
        && Option.isNone(state.content.operation),
    ))
    expect(choosing._tag).toBe("Open")
    harness.registry.dispose()
  })

  it("installs before assignment for an uninstalled choice", async () => {
    const harness = makeHarness({ installed: false })
    await Effect.runPromise(execute(harness.registry, harness.service.select, providerModelId))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Open" && state.content._tag === "Harness"))
    await Effect.runPromise(execute(harness.registry, harness.service.continueWithMagnitude, undefined))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))

    const mutations = harness.calls.filter((call) => [
      "SyncLocalModel",
      "AssignModelSlot",
      "LoadModelSlot",
      "CompleteOnboarding",
    ].includes(call))
    expect(mutations).toEqual([
      "SyncLocalModel",
      "AssignModelSlot",
      "LoadModelSlot",
      "CompleteOnboarding",
    ])
    harness.registry.dispose()
  })

  it("preserves a structured download failure as the terminal setup failure", async () => {
    const failure: ModelAcquisitionFailure = {
      _tag: "InsufficientDiskSpace",
      requiredBytes: 40,
      availableBytes: 30,
    }
    const harness = makeHarness({ installed: false, downloadFailure: failure })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      providerModelId,
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
    expect(Option.getOrThrow(state.notice)).toEqual({
      failure,
      subject: {
        _tag: "ModelOperation",
        operation: "Installing",
        model: makeModel(false),
      },
    })
    expect(harness.calls).not.toContain("AssignModelSlot")
    expect(harness.calls).not.toContain("LoadModelSlot")
    harness.registry.dispose()
  })

  it("does not mistake a pre-existing download provider identity for installation", async () => {
    const harness = makeHarness({ installed: false, initiallyDownloading: true })
    await Effect.runPromise(execute(harness.registry, harness.service.select, providerModelId))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Open" && state.content._tag === "Harness"))
    await Effect.runPromise(execute(harness.registry, harness.service.continueWithMagnitude, undefined))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))

    expect(harness.calls.indexOf("SyncLocalModel")).toBeGreaterThanOrEqual(0)
    expect(harness.calls.indexOf("SyncLocalModel")).toBeLessThan(harness.calls.indexOf("AssignModelSlot"))
    harness.registry.dispose()
  })

  it("stops after a failed dependency and never runs downstream mutations", async () => {
    const harness = makeHarness({ installed: true, failAssign: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      providerModelId,
    ))

    const failed = await Effect.runPromise(waitForView(
      harness,
      (view) => view._tag === "Open"
        && view.content._tag === "Chooser"
        && Option.isSome(view.notice),
    ))
    if (failed._tag !== "Open") throw new Error("expected open setup")
    expect(onboardingModelSetupNoticeMessage(Option.getOrThrow(failed.notice)))
      .toBe("Unexpected error configuring Setup Model (Q4) · assignment rejected")
    expect(harness.calls).toContain("AssignModelSlot")
    expect(harness.calls).not.toContain("LoadModelSlot")
    expect(harness.calls).not.toContain("CompleteOnboarding")
    harness.registry.dispose()
  })

  it("does not load through a slot selection replacement", async () => {
    const harness = makeHarness({ installed: true, replaceSelectionBeforeLoad: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      providerModelId,
    ))

    await Effect.runPromise(waitForView(
      harness,
      (view) => view._tag === "Open"
        && view.content._tag === "Chooser"
        && Option.isSome(view.notice),
    ))
    expect(harness.calls).toContain("LoadModelSlot")
    expect(harness.calls).not.toContain("CompleteOnboarding")
    expect(harness.onboardingCompleted()).toBe(false)
    harness.registry.dispose()
  })

  it("cancels only the exact admitted download and waits for its terminal fact", async () => {
    const harness = makeHarness({ installed: false, keepDownloading: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.select,
      providerModelId,
    ))
    await Effect.runPromise(waitForCall(harness.calls, "SyncLocalModel"))
    await Effect.runPromise(execute(harness.registry, harness.service.cancel, undefined))

    expect(harness.cancelledDownloads).toEqual([providerModelId])
    expect(harness.calls).not.toContain("AssignModelSlot")
    expect(harness.calls).not.toContain("LoadModelSlot")
    harness.registry.dispose()
  })

  it("detaches from an admitted load without stopping the shared instance", async () => {
    const harness = makeHarness({ installed: true, keepLoading: true })
    await Effect.runPromise(execute(harness.registry, harness.service.select, providerModelId))
    await Effect.runPromise(waitForCall(harness.calls, "LoadModelSlot"))

    await Effect.runPromise(execute(harness.registry, harness.service.cancel, undefined))

    expect(harness.calls).not.toContain("StopModelSlot")
    expect(harness.stoppedInstances).toEqual([])
    expect(harness.onboardingCompleted()).toBe(false)
    harness.registry.dispose()
  })

  it("exiting required setup completes onboarding without touching any model resource", async () => {
    const harness = makeHarness({ installed: false })
    await Effect.runPromise(execute(harness.registry, harness.service.exit, undefined))
    await Effect.runPromise(waitForView(harness, (state) => state._tag === "Closed"))

    expect(harness.onboardingCompleted()).toBe(true)
    expect(harness.calls.filter((call) => [
      "SyncLocalModel",
      "AssignModelSlot",
      "LoadModelSlot",
      "StopModelSlot",
      "CancelLocalModelSync",
    ].includes(call))).toEqual([])
    harness.registry.dispose()
  })

  it("keeps active setup alive when its React-style observer unmounts", async () => {
    const harness = makeHarness({ installed: true, keepLoading: true })
    const unmount = harness.registry.mount(harness.service.view)
    await Effect.runPromise(execute(harness.registry, harness.service.select, providerModelId))
    await Effect.runPromise(waitForCall(harness.calls, "LoadModelSlot"))
    unmount()
    await Effect.runPromise(Effect.sleep("5 millis"))

    expect(harness.calls).not.toContain("CompleteOnboarding")
    await Effect.runPromise(execute(harness.registry, harness.service.cancel, undefined))
    expect(harness.stoppedInstances).toEqual([])
    harness.registry.dispose()
  })

})
