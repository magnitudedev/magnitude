import { Atom, Registry, Result } from "@effect-atom/atom-react"
import { Cause, Effect, Fiber, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import { Client as EffectQueryClient } from "@magnitudedev/effect-query"
import {
  AcnRpcClientTag,
  AssessmentEnvironmentIdSchema,
  CatalogModelIdSchema,
  CatalogVariantIdSchema,
  ModelDownloadIdSchema,
  ModelAssessmentIdSchema,
  ModelInstanceIdSchema,
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  ModelSlotConfiguredLocal,
  ModelSlotUnassigned,
  ModelVariantLabelSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  type AcnRpcClient,
  type LocalModel,
  type ModelDownloadFailure,
  type LocalModelsState,
  type ModelSlotsState,
  type SlotSelection,
} from "@magnitudedev/sdk"
import { clientServicesLayer, type ClientServices } from "../state/client-services"
import { localModelProviderModelId } from "./projection"
import { installationAdmissionIsVisible } from "./service"
import { OnboardingModelSetup } from "./setup"

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
        maximumContextLength: 32_768,
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
      quantization: "Q4_K_M",
      precisionLabel: "4-bit",
    },
    downloadBytes: 1,
    catalogMembershipState: {
      _tag: "InCatalog",
      catalogData: {
        modelId: CatalogModelIdSchema.make("setup-model"),
        variantId: CatalogVariantIdSchema.make("gguf:q4"),
        intelligenceScore: 1,
        intelligenceScoreSource: "test",
        fidelityRank: 1,
        quantizationAware: false,
        qualityNotes: [],
      },
    },
    upgradeState: installed ? { _tag: "Current" } : { _tag: "NotApplicable" },
    acquisitionState: installed
      ? { _tag: "Installed", installedBytes: 1, origins: ["Magnitude"] }
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
      recommendations: [],
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
      instance: lifecycle === "None" ? Option.none() : Option.some({
        id,
        configurationId,
        lifecycle: lifecycle === "Ready"
          ? { _tag: "Ready", allocation }
          : lifecycle === "Stopped"
            ? { _tag: "Stopped", reason: "user_stop" }
            : {
                _tag: "Loading",
                stage: "loading",
                progress: Option.none(),
                plannedAllocation: Option.none(),
              },
      }),
      actions: lifecycle === "None" || lifecycle === "Stopped" ? ["Load"] : ["Stop"],
    }),
  },
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
  readonly initiallyDownloading?: boolean
  readonly ready?: boolean
  readonly keepLoading?: boolean
  readonly keepDownloading?: boolean
  readonly failAssign?: boolean
  readonly replaceLoadInstance?: boolean
  readonly keepCompleting?: boolean
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
  let onboardingCompleted = false
  let revision = 0
  const calls: string[] = []
  const stoppedInstances: unknown[] = []
  const cancelledDownloads: unknown[] = []
  const rpc = ((name: string, payload: any) => Effect.suspend(() => {
    calls.push(name)
    switch (name) {
      case "WatchMirroredStates": return Effect.succeed(Stream.never)
      case "GetLocalModels": return Effect.succeed({ revision: revision++, state: models })
      case "GetModelSlots": return Effect.succeed({ revision: revision++, state: slots })
      case "GetOnboardingState": return Effect.succeed({
        revision: revision++,
        state: { completed: onboardingCompleted },
      })
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
          options.replaceLoadInstance
            ? ModelInstanceIdSchema.make("replacement-instance")
            : instanceId,
        )
        return Effect.succeed({ instanceId })
      case "StopModel":
        stoppedInstances.push(payload.instanceId)
        slots = configuredSlots("Stopped", payload.instanceId)
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
      case "UpdateOnboardingState":
        if (options.keepCompleting) return Effect.never
        onboardingCompleted = payload.completed
        return Effect.succeed({})
      default: return Effect.die(new Error(`Unexpected RPC ${name}`))
    }
  })) as unknown as AcnRpcClient
  const effectQuery = EffectQueryClient.make<AcnRpcClientTag, never, ClientServices, never>(
    Layer.succeed(AcnRpcClientTag, rpc),
    (client) => clientServicesLayer(client),
  )
  const registry = Registry.make()
  const serviceReference = effectQuery.runtime.atom(OnboardingModelSetup)
  const service = {
    state: Atom.make((get) => Result.flatMap(
      get(serviceReference),
      (setup) => get(setup.state),
    )),
    start: Atom.keepAlive(effectQuery.runtime.fn<typeof configurationId>()(
      (input) => Effect.flatMap(OnboardingModelSetup, (setup) => setup.start(input)),
      { concurrent: true },
    )),
    cancel: Atom.keepAlive(effectQuery.runtime.fn(
      () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.cancel),
      { concurrent: true },
    )),
    skip: Atom.keepAlive(effectQuery.runtime.fn(
      () => Effect.flatMap(OnboardingModelSetup, (setup) => setup.skip),
      { concurrent: true },
    )),
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

describe("OnboardingModelSetupService", () => {
  it("is observational and passive across state remounts", async () => {
    const harness = makeHarness({ installed: true, ready: true })
    const firstUnmount = harness.registry.mount(harness.service.state)
    await Effect.runPromise(Effect.sleep("5 millis"))
    firstUnmount()
    const secondUnmount = harness.registry.mount(harness.service.state)
    await Effect.runPromise(Effect.sleep("5 millis"))
    const state = Result.value(harness.registry.get(harness.service.state))
    expect(Option.isSome(state) && state.value._tag).toBe("Choosing")
    secondUnmount()

    expect(harness.calls.filter((call) => [
      "ReconcileCatalogModel",
      "AssignSlot",
      "LoadModel",
      "StopModel",
      "CancelModelDownload",
      "UpdateOnboardingState",
    ].includes(call))).toEqual([])
    harness.registry.dispose()
  })

  it("only completes onboarding for an exact already-ready choice", async () => {
    const harness = makeHarness({ installed: true, ready: true })
    await Effect.runPromise(execute(
      harness.registry,
      harness.service.start,
      configurationId,
    ))

    expect(harness.onboardingCompleted()).toBe(true)
    expect(harness.calls).not.toContain("ReconcileCatalogModel")
    expect(harness.calls).not.toContain("AssignSlot")
    expect(harness.calls).not.toContain("LoadModel")
    harness.registry.dispose()
  })

  it("does not admit cancellation once an already-ready choice is completing", async () => {
    const harness = makeHarness({ installed: true, ready: true, keepCompleting: true })
    const start = Effect.runFork(execute(
      harness.registry,
      harness.service.start,
      configurationId,
    ))
    await Effect.runPromise(waitForCall(harness.calls, "UpdateOnboardingState"))

    const cancellation = await Effect.runPromiseExit(execute(
      harness.registry,
      harness.service.cancel,
      undefined,
    ))
    expect(cancellation._tag).toBe("Failure")
    if (cancellation._tag === "Failure") {
      expect(Cause.pretty(cancellation.cause)).toContain("OnboardingModelSetupCancellationUnavailable")
    }
    await Effect.runPromise(Fiber.interrupt(start))
    harness.registry.dispose()
  })

  it("assigns, loads, awaits, and completes an installed non-ready choice", async () => {
    const harness = makeHarness({ installed: true })
    await Effect.runPromise(execute(harness.registry, harness.service.start, configurationId))

    const mutations = harness.calls.filter((call) => [
      "ReconcileCatalogModel",
      "AssignSlot",
      "LoadModel",
      "UpdateOnboardingState",
    ].includes(call))
    expect(mutations).toEqual(["AssignSlot", "LoadModel", "UpdateOnboardingState"])
    harness.registry.dispose()
  })

  it("installs before assignment for an uninstalled choice", async () => {
    const harness = makeHarness({ installed: false })
    await Effect.runPromise(execute(harness.registry, harness.service.start, configurationId))

    const mutations = harness.calls.filter((call) => [
      "ReconcileCatalogModel",
      "AssignSlot",
      "LoadModel",
      "UpdateOnboardingState",
    ].includes(call))
    expect(mutations).toEqual([
      "ReconcileCatalogModel",
      "AssignSlot",
      "LoadModel",
      "UpdateOnboardingState",
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
    const exit = await Effect.runPromiseExit(execute(
      harness.registry,
      harness.service.start,
      configurationId,
    ))

    expect(exit._tag).toBe("Failure")
    const state = Option.getOrThrow(Result.value(harness.registry.get(harness.service.state)))
    expect(state).toMatchObject({
      _tag: "Failed",
      failure,
    })
    expect(harness.calls).not.toContain("AssignSlot")
    expect(harness.calls).not.toContain("LoadModel")
    harness.registry.dispose()
  })

  it("does not mistake a pre-existing download provider identity for installation", async () => {
    const harness = makeHarness({ installed: false, initiallyDownloading: true })
    await Effect.runPromise(execute(harness.registry, harness.service.start, configurationId))

    expect(harness.calls.indexOf("ReconcileCatalogModel")).toBeGreaterThanOrEqual(0)
    expect(harness.calls.indexOf("ReconcileCatalogModel")).toBeLessThan(harness.calls.indexOf("AssignSlot"))
    harness.registry.dispose()
  })

  it("stops after a failed dependency and never runs downstream mutations", async () => {
    const harness = makeHarness({ installed: true, failAssign: true })
    const exit = await Effect.runPromiseExit(execute(
      harness.registry,
      harness.service.start,
      configurationId,
    ))

    expect(exit._tag).toBe("Failure")
    expect(harness.calls).toContain("AssignSlot")
    expect(harness.calls).not.toContain("LoadModel")
    expect(harness.calls).not.toContain("UpdateOnboardingState")
    harness.registry.dispose()
  })

  it("does not load through a slot selection replacement", async () => {
    const harness = makeHarness({ installed: true, replaceSelectionBeforeLoad: true })
    const exit = await Effect.runPromiseExit(execute(
      harness.registry,
      harness.service.start,
      configurationId,
    ))

    expect(exit._tag).toBe("Failure")
    expect(harness.calls).toContain("LoadModel")
    expect(harness.calls).not.toContain("UpdateOnboardingState")
    expect(harness.onboardingCompleted()).toBe(false)
    harness.registry.dispose()
  })

  it("cancels only the exact admitted download and waits for its terminal fact", async () => {
    const harness = makeHarness({ installed: false, keepDownloading: true })
    const start = Effect.runFork(execute(
      harness.registry,
      harness.service.start,
      configurationId,
    ))
    await Effect.runPromise(waitForCall(harness.calls, "ReconcileCatalogModel"))
    await Effect.runPromise(execute(harness.registry, harness.service.cancel, undefined))

    await Effect.runPromise(Fiber.join(start))
    expect(harness.cancelledDownloads).toEqual([downloadId])
    expect(harness.calls).not.toContain("AssignSlot")
    expect(harness.calls).not.toContain("LoadModel")
    harness.registry.dispose()
  })

  it("skip completes onboarding without touching any model resource", async () => {
    const harness = makeHarness({ installed: false })
    await Effect.runPromise(execute(harness.registry, harness.service.skip, undefined))

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
    const unmount = harness.registry.mount(harness.service.start)
    harness.registry.set(harness.service.start, configurationId)
    await Effect.runPromise(waitForCall(harness.calls, "LoadModel"))
    unmount()
    await Effect.runPromise(Effect.sleep("5 millis"))

    expect(harness.calls).not.toContain("UpdateOnboardingState")
    await Effect.runPromise(execute(harness.registry, harness.service.cancel, undefined))
    expect(harness.stoppedInstances).toEqual([instanceId])
    harness.registry.dispose()
  })

  it("never accepts a replacement instance as the admitted instance", async () => {
    const harness = makeHarness({ installed: true, replaceLoadInstance: true })
    const exit = await Effect.runPromiseExit(execute(
      harness.registry,
      harness.service.start,
      configurationId,
    ))
    expect(exit._tag).toBe("Failure")
    expect(harness.onboardingCompleted()).toBe(false)
    expect(harness.calls).not.toContain("UpdateOnboardingState")
    harness.registry.dispose()
  })
})
