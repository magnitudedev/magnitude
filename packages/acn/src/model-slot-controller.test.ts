import { describe, expect, it } from "vitest"
import {
  Deferred,
  Effect,
  Fiber,
  Layer,
  Option,
  Ref,
  Schema,
  Stream,
  SubscriptionRef,
} from "effect"
import {
  CatalogCandidateIdSchema,
  ModelOfferingTargetIdSchema,
  ModelFileIdSchema,
  ModelPackageIdSchema,
  RecommendableModelIdSchema,
  ModelServingConfigurationIdSchema,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  ProviderModelCatalogReady,
  type LocalProviderOffering,
  type ProviderModelCatalogEntry,
  type SlotSelection,
} from "@magnitudedev/protocol"
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/sdk"
import { resolveContextLimitPolicy } from "@magnitudedev/storage"
import {
  IcnClient,
  IcnInstances,
  Generated,
  type IcnClientService,
} from "@magnitudedev/icn"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { ModelConfiguration } from "./model-configuration"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelRecommendations } from "./local-model-recommendations"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { ProviderModelCatalog } from "./provider-model-catalog"
import { MirroredStateChangesLive } from "./mirrored-state"
import {
  ModelSlotController,
  ModelSlotControllerLive,
} from "./model-slot-controller"

const packageId = ModelPackageIdSchema.make("test-package")
const configurationId = ModelServingConfigurationIdSchema.make("test-configuration")
const providerModelId = ProviderModelIdSchema.make(configurationId)
const modelPackage = {
  id: packageId,
  source: {
    _tag: "HuggingFace" as const,
    repository: "test/model",
    revision: "a".repeat(40),
  },
  files: [{
    id: ModelFileIdSchema.make("weights"),
    path: "model.gguf",
    role: "weights" as const,
    sizeBytes: 1,
    sha256: "a".repeat(64),
  }],
  relationships: [],
  properties: {
    format: "gguf",
    quantization: "Q4",
    quantizationName: "4-bit",
    architecture: "dense",
    maximumContextLength: 8_192,
  },
}
const selection: SlotSelection = {
  providerId: LOCAL_PROVIDER_ID,
  providerModelId,
  reasoningEffort: ReasoningEffortSchema.make("none"),
}
const capabilities = {
  vision: false,
  tools: true,
  structuredOutput: true,
  reasoning: {
    supported: false,
    efforts: [],
    defaultEffort: Option.none(),
  },
}
const catalogModel: ProviderModelCatalogEntry = {
  providerId: LOCAL_PROVIDER_ID,
  providerModelId,
  modelFamilyId: Option.none(),
  displayName: "Test local model",
  supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
  contextWindow: 8_192,
  maxOutputTokens: 1_024,
  memory: Option.none(),
  capabilities,
  availability: { _tag: "Available" },
  pricing: Option.none(),
}
const offering = {
  providerModelId,
  targetId: ModelOfferingTargetIdSchema.make("test-model"),
  configuration: {
    id: configurationId,
    target: {
      _tag: "Package",
      package: modelPackage,
    },
    profile: { contextLength: 8_192 },
  },
  origin: { _tag: "UserConfigured" },
  capabilities,
} as LocalProviderOffering

const recommendationEntry = {
  candidate: {
    id: CatalogCandidateIdSchema.make("test-configuration"),
    targetId: offering.targetId,
    providerModelId,
    displayName: "Test local model",
    description: "Test model",
    license: "test",
    profile: offering.configuration.profile,
    downloadBytes: 1,
    download: { _tag: "Downloaded" as const, installedBytes: 1 },
    preparation: { _tag: "Available" as const, providerModelIds: [providerModelId] },
    quantization: "Q4",
    quantizationName: "4-bit",
    memory: [],
    intelligenceScore: 0,
    intelligenceProvenance: "Unavailable",
    fidelityRank: 0,
    qualityEvidence: [],
    estimatedTokensPerSecond: Option.none(),
    capabilities,
    sources: [{
      source: modelPackage.source,
      files: modelPackage.files.map(({ path, sha256 }) => ({ path, sha256 })),
    }],
  },
  recommendableModelId: RecommendableModelIdSchema.make("test-model"),
  configuration: offering.configuration,
  recommendation: Option.none(),
}

const makeHarness = (options: {
  readonly initiallyAssigned?: boolean
  readonly initialOfferings?: readonly LocalProviderOffering[]
  readonly exposeRecommendation?: boolean
  readonly installed?: boolean
  readonly catalogAvailability?: ProviderModelCatalogEntry["availability"]
} = {}) => Effect.gen(function* () {
  const configuration = yield* SubscriptionRef.make({
    slots: {
      primary: options.initiallyAssigned === false
        ? Option.none<SlotSelection>()
        : Option.some(selection),
      secondary: Option.none<SlotSelection>(),
    },
    localModelRecency: { primary: [providerModelId], secondary: [] },
    favoriteModels: [],
    localProviderOfferings: [],
    dismissedDownloadFailures: [],
    contextLimits: resolveContextLimitPolicy({ onboarding: Option.none() }),
  })
  const instances = yield* SubscriptionRef.make<Generated.ModelInstancesSnapshot>({
    revision: 0,
    instances: [],
  })
  const catalogSnapshot = yield* SubscriptionRef.make({
    revision: 0,
    state: new ProviderModelCatalogReady({
      providers: [{
        providerId: ProviderIdSchema.make("local"),
        displayName: "Local",
        authentication: "NotRequired" as const,
        availability: { _tag: "Available" as const },
      }],
      models: [{
        ...catalogModel,
        availability: options.catalogAvailability ?? catalogModel.availability,
      }],
    }),
  })
  const loadCalls = yield* Ref.make(0)
  const previewCalls = yield* Ref.make(0)
  const stopCalls = yield* Ref.make<readonly string[]>([])
  const offerings = yield* Ref.make<readonly LocalProviderOffering[]>(
    options.initialOfferings ?? [offering],
  )
  const loadEntered = yield* Deferred.make<void>()
  const releaseLoad = yield* Deferred.make<void>()

  const client = {
    models: {
      previewModelLoad: () => Effect.gen(function* () {
        yield* Ref.update(previewCalls, (count) => count + 1)
        return {
          contextWindowTokens: 8_192,
          parallelSequences: 1,
          physicalContextTokens: 8_192,
          requiredSystemMemoryBytes: 1,
        }
      }),
      loadModelInstance: ({ payload }: {
        readonly payload: {
          readonly instanceId: string
          readonly configuration: { readonly id: string }
        }
      }) => Effect.gen(function* () {
        yield* Ref.update(loadCalls, (count) => count + 1)
        yield* Deferred.succeed(loadEntered, undefined)
        yield* Deferred.await(releaseLoad)
        const instance = yield* Schema.decodeUnknown(Generated.ModelInstance)({
          id: payload.instanceId,
          configurationId: payload.configuration.id,
          lifecycle: {
            _tag: "Loading",
            stage: "queued",
          },
        })
        yield* SubscriptionRef.update(instances, (current) => ({
          revision: current.revision + 1,
          instances: [instance],
        }))
        return { status: 200, headers: {}, events: Stream.empty }
      }),
      stopModelInstance: ({ path }: { readonly path: { readonly instance_id: string } }) =>
        Ref.update(stopCalls, (ids) => [...ids, path.instance_id]).pipe(
          Effect.as({}),
        ),
    },
  } as unknown as IcnClientService

  const dependencies = Layer.mergeAll(
    Layer.succeed(ModelConfiguration, ModelConfiguration.of({
      get: SubscriptionRef.get(configuration),
      changes: configuration.changes,
      updateSlot: (slotId, next) => SubscriptionRef.update(configuration, (current) => ({
        ...current,
        slots: { ...current.slots, [slotId]: next },
      })),
      recordUse: () => Effect.void,
      setFavorite: () => Effect.void,
    })),
    Layer.succeed(LocalModelPackages, LocalModelPackages.of({
      snapshot: Effect.succeed({ revision: 0, state: { entries: [] } }),
      changes: Stream.empty,
      installedPackageIds: Effect.succeed(
        new Set(options.installed === false ? [] : [packageId]),
      ),
      acquireTarget: () => Effect.void,
      cancelTargetDownload: () => Effect.void,
      dismissTargetFailure: () => Effect.void,
      removeTargetPackages: () => Effect.void,
    })),
    Layer.succeed(LocalModelRecommendations, LocalModelRecommendations.of({
      snapshot: Effect.succeed({
        revision: 0,
        state: { _tag: "Ready", recommendations: [], catalog: [], progress: [] },
      }),
      changes: Stream.empty,
      refresh: Effect.void,
      getCatalogByProviderModelId: (requestedProviderModelId) => Effect.succeed(
        options.exposeRecommendation && requestedProviderModelId === providerModelId
          ? recommendationEntry
          : undefined,
      ),
    })),
    Layer.succeed(LocalProviderOfferings, LocalProviderOfferings.of({
      list: Ref.get(offerings),
      changes: Stream.empty,
      resolve: (requestedProviderModelId) => Ref.get(offerings).pipe(
        Effect.flatMap((current) => {
          const resolved = current.find((item) =>
            item.providerModelId === requestedProviderModelId)
          return resolved
            ? Effect.succeed(resolved)
            : Effect.die("Offering was not retained")
        }),
      ),
      save: (targetId, savedConfiguration, origin) => {
        const saved = {
          ...offering,
          providerModelId: ProviderModelIdSchema.make(savedConfiguration.id),
          targetId,
          configuration: savedConfiguration,
          origin,
        }
        return Ref.update(offerings, (current) => [...current, saved]).pipe(
          Effect.as(saved),
        )
      },
    })),
    Layer.succeed(ProviderModelCatalog, ProviderModelCatalog.of({
      snapshot: SubscriptionRef.get(catalogSnapshot),
      changes: catalogSnapshot.changes,
      refresh: () => Effect.void,
    })),
    Layer.succeed(IcnClient, client),
    Layer.succeed(IcnInstances, IcnInstances.of({
      get: SubscriptionRef.get(instances),
      changes: instances.changes,
      initialized: Effect.succeed(true),
      refresh: Effect.void,
    })),
    MirroredStateChangesLive,
  )

  return {
    layer: ModelSlotControllerLive.pipe(Layer.provide(dependencies)),
    loadCalls,
    previewCalls,
    stopCalls,
    offerings,
    instances,
    catalogSnapshot,
    loadEntered,
    releaseLoad,
  }
})

const releaseLoadAsReady = (
  harness: Effect.Effect.Success<ReturnType<typeof makeHarness>>,
) => Effect.gen(function* () {
  const loading = yield* harness.instances.changes.pipe(
    Stream.filter((snapshot) =>
      snapshot.instances.some((instance) => instance.lifecycle._tag === "Loading")),
    Stream.runHead,
    Effect.fork,
  )
  yield* Deferred.succeed(harness.releaseLoad, undefined)
  const loadingSnapshot = yield* Fiber.join(loading).pipe(
    Effect.flatMap(Option.match({
      onNone: () => Effect.die("Model instance stream ended before Loading"),
      onSome: Effect.succeed,
    })),
  )
  const instance = loadingSnapshot.instances[0]
  if (!instance) return yield* Effect.die("Loading instance was absent")
  yield* SubscriptionRef.set(harness.instances, {
    revision: loadingSnapshot.revision + 1,
    instances: [{
      ...instance,
      lifecycle: {
        _tag: "Ready",
        allocation: {
          contextWindowTokens: 8_192,
          parallelSequences: 1,
          physicalContextTokens: 8_192,
          memoryDomains: [],
        },
      },
    }],
  })
})

describe("ModelSlotController load admission", () => {
  it("rejects an uninstalled assignment without changing the slot", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness({
        initiallyAssigned: false,
        installed: false,
      })
      yield* Effect.gen(function* () {
        const controller = yield* ModelSlotController
        const outcome = yield* Effect.exit(controller.updateModelSlot(
          PRIMARY_SLOT_ID,
          Option.some(selection),
        ))
        expect(outcome._tag).toBe("Failure")
        expect((yield* controller.snapshot).state.slots.primary._tag).toBe("Unassigned")
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("retains a recommendation configuration through ordinary slot assignment", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness({
        initiallyAssigned: false,
        initialOfferings: [],
        exposeRecommendation: true,
      })
      yield* Effect.gen(function* () {
        const controller = yield* ModelSlotController
        yield* controller.updateModelSlot(
          PRIMARY_SLOT_ID,
          Option.some(selection),
        )

        expect(yield* Ref.get(harness.offerings)).toEqual([offering])
        expect((yield* controller.snapshot).state.slots.primary).toMatchObject({
          _tag: "ConfiguredLocal",
          selection,
          availability: { _tag: "Available" },
        })
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("makes an immediately following load admissible when assignment succeeds", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness({ initiallyAssigned: false })
      yield* Effect.gen(function* () {
        const controller = yield* ModelSlotController
        yield* controller.updateModelSlot(
          PRIMARY_SLOT_ID,
          Option.some(selection),
        )

        const assigned = (yield* controller.snapshot).state.slots.primary
        expect(assigned).toMatchObject({
          _tag: "ConfiguredLocal",
          availability: { _tag: "Available" },
          actions: ["Load"],
        })

        const loading = yield* controller.loadModel(PRIMARY_SLOT_ID).pipe(Effect.fork)
        yield* Deferred.await(harness.loadEntered)
        yield* releaseLoadAsReady(harness)
        yield* Fiber.join(loading)
        expect(yield* Ref.get(harness.loadCalls)).toBe(1)
        expect(yield* Ref.get(harness.previewCalls)).toBe(0)
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("does not gate an installed exact configuration on a stale catalog projection", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness({
        initiallyAssigned: false,
        catalogAvailability: {
          _tag: "Disabled",
          reason: "installation_unavailable",
        },
      })
      yield* Effect.gen(function* () {
        const controller = yield* ModelSlotController
        yield* controller.updateModelSlot(
          PRIMARY_SLOT_ID,
          Option.some(selection),
        )

        expect((yield* controller.snapshot).state.slots.primary).toMatchObject({
          _tag: "ConfiguredLocal",
          availability: { _tag: "Available" },
          actions: ["Load"],
        })
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("previews only when explicitly requested", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness()
      yield* Effect.gen(function* () {
        const controller = yield* ModelSlotController
        expect(yield* Ref.get(harness.previewCalls)).toBe(0)
        expect(yield* controller.previewModelLoad(PRIMARY_SLOT_ID)).toMatchObject({
          contextWindowTokens: 8_192,
          parallelSequences: 1,
        })
        expect(yield* Ref.get(harness.previewCalls)).toBe(1)
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("submits one native instance for concurrent equivalent load commands", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness()
      yield* Effect.gen(function* () {
        const controller = yield* ModelSlotController
        expect((yield* controller.snapshot).state.slots.primary).toMatchObject({
          _tag: "ConfiguredLocal",
          actions: ["Load"],
        })
        const first = yield* controller.loadModel(PRIMARY_SLOT_ID).pipe(Effect.fork)
        const second = yield* controller.loadModel(PRIMARY_SLOT_ID).pipe(Effect.fork)
        const admission = yield* Effect.race(
          Deferred.await(harness.loadEntered).pipe(Effect.as({ _tag: "Entered" as const })),
          Fiber.await(first).pipe(Effect.map((exit) => ({ _tag: "Finished" as const, exit }))),
        )
        expect(admission._tag).toBe("Entered")
        if (admission._tag !== "Entered") return
        expect(yield* Ref.get(harness.loadCalls)).toBe(1)
        yield* releaseLoadAsReady(harness)
        yield* Fiber.join(first)
        yield* Fiber.join(second)
        expect(yield* Ref.get(harness.loadCalls)).toBe(1)
        expect(yield* Ref.get(harness.previewCalls)).toBe(0)
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("does not hold command admission across transport and stops a superseded exact instance", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness()
      yield* Effect.gen(function* () {
        const controller = yield* ModelSlotController
        const loading = yield* controller.loadModel(PRIMARY_SLOT_ID).pipe(Effect.fork)
        yield* Deferred.await(harness.loadEntered)

        yield* controller.updateModelSlot(PRIMARY_SLOT_ID, Option.none())
        expect((yield* controller.snapshot).state.slots.primary._tag).toBe("Unassigned")

        yield* Deferred.succeed(harness.releaseLoad, undefined)
        const outcome = yield* Fiber.await(loading)
        expect(outcome._tag).toBe("Failure")
        expect(yield* Ref.get(harness.stopCalls)).toHaveLength(1)
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("shares one physical command across slots selecting the same configuration", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness()
      yield* Effect.gen(function* () {
        const controller = yield* ModelSlotController
        yield* controller.updateModelSlot(SECONDARY_SLOT_ID, Option.some(selection))
        expect((yield* controller.snapshot).state.slots.secondary).toMatchObject({
          _tag: "ConfiguredLocal",
          actions: ["Load"],
        })

        const primary = yield* controller.loadModel(PRIMARY_SLOT_ID).pipe(Effect.fork)
        const secondary = yield* controller.loadModel(SECONDARY_SLOT_ID).pipe(Effect.fork)
        yield* Deferred.await(harness.loadEntered)
        expect(yield* Ref.get(harness.loadCalls)).toBe(1)

        yield* releaseLoadAsReady(harness)
        yield* Fiber.join(primary)
        yield* Fiber.join(secondary)
        expect(yield* Ref.get(harness.loadCalls)).toBe(1)
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("waits for the exact selected instance to become ready", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness()
      yield* Effect.gen(function* () {
        const controller = yield* ModelSlotController
        const ready = yield* controller.ensureLocalModelReady(
          PRIMARY_SLOT_ID,
          providerModelId,
        ).pipe(Effect.fork)
        yield* Deferred.await(harness.loadEntered)
        const loading = yield* harness.instances.changes.pipe(
          Stream.filter((snapshot) =>
            snapshot.instances.some((instance) => instance.lifecycle._tag === "Loading")),
          Stream.runHead,
          Effect.fork,
        )
        yield* Deferred.succeed(harness.releaseLoad, undefined)
        const loadingSnapshot = yield* Fiber.join(loading).pipe(
          Effect.flatMap(Option.match({
            onNone: () => Effect.die("Model instance stream ended before Loading"),
            onSome: Effect.succeed,
          })),
        )
        const instance = loadingSnapshot.instances[0]
        if (!instance) return yield* Effect.die("Loading instance was absent")

        yield* SubscriptionRef.set(harness.instances, {
          revision: loadingSnapshot.revision + 1,
          instances: [{
            ...instance,
            lifecycle: {
              _tag: "Ready",
              allocation: {
                contextWindowTokens: 8_192,
                parallelSequences: 1,
                physicalContextTokens: 8_192,
                memoryDomains: [],
              },
            },
          }],
        })
        yield* Fiber.join(ready)
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("returns cancellation for an explicit stop and binds retry to the replacement instance", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness()
      yield* Effect.gen(function* () {
        const controller = yield* ModelSlotController
        const firstLoad = yield* controller.loadModel(PRIMARY_SLOT_ID).pipe(Effect.fork)
        yield* Deferred.await(harness.loadEntered)
        const firstLoading = yield* harness.instances.changes.pipe(
          Stream.filter((snapshot) =>
            snapshot.instances.some((instance) => instance.lifecycle._tag === "Loading")),
          Stream.runHead,
          Effect.fork,
        )
        yield* Deferred.succeed(harness.releaseLoad, undefined)
        const firstSnapshot = yield* Fiber.join(firstLoading)
        expect(Option.isSome(firstSnapshot)).toBe(true)
        if (Option.isNone(firstSnapshot)) return
        const firstInstance = firstSnapshot.value.instances[0]
        expect(firstInstance).toBeDefined()
        if (firstInstance === undefined) return

        yield* SubscriptionRef.set(harness.instances, {
          revision: firstSnapshot.value.revision + 1,
          instances: [{
            ...firstInstance,
            lifecycle: { _tag: "Stopped", reason: "user_stop" },
          }],
        })
        expect(yield* Fiber.join(firstLoad)).toEqual({
          _tag: "Cancelled",
          instanceId: firstInstance.id,
          reason: "user_stop",
        })

        const replacementLoading = yield* harness.instances.changes.pipe(
          Stream.filter((snapshot) => snapshot.instances.some((instance) =>
            instance.id !== firstInstance.id && instance.lifecycle._tag === "Loading")),
          Stream.runHead,
          Effect.fork,
        )
        const retry = yield* controller.loadModel(PRIMARY_SLOT_ID).pipe(Effect.fork)
        const replacementSnapshot = yield* Fiber.join(replacementLoading)
        expect(Option.isSome(replacementSnapshot)).toBe(true)
        if (Option.isNone(replacementSnapshot)) return
        const replacement = replacementSnapshot.value.instances[0]
        expect(replacement).toBeDefined()
        if (replacement === undefined) return
        expect(replacement.id).not.toBe(firstInstance.id)

        yield* SubscriptionRef.set(harness.instances, {
          revision: replacementSnapshot.value.revision + 1,
          instances: [{
            ...replacement,
            lifecycle: {
              _tag: "Ready",
              allocation: {
                contextWindowTokens: 8_192,
                parallelSequences: 1,
                physicalContextTokens: 8_192,
                memoryDomains: [],
              },
            },
          }],
        })
        expect(yield* Fiber.join(retry)).toMatchObject({
          _tag: "Ready",
          instanceId: replacement.id,
        })
        expect(yield* Ref.get(harness.loadCalls)).toBe(2)
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("publishes catalog-derived agent configuration changes even when slot identity is unchanged", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness()
      yield* Effect.gen(function* () {
        const controller = yield* ModelSlotController
        const before = yield* controller.agentModelConfiguration

        yield* SubscriptionRef.update(harness.catalogSnapshot, (current) => ({
          revision: current.revision + 1,
          state: new ProviderModelCatalogReady({
            providers: current.state.providers,
            models: [{
              ...catalogModel,
              contextWindow: 16_384,
              maxOutputTokens: 2_048,
            }],
          }),
        }))

        const changed = yield* controller.agentModelConfigurations.pipe(
          Stream.filter((configuration) => configuration.revision > before.revision),
          Stream.runHead,
          Effect.flatMap(Option.match({
            onNone: () => Effect.die("Agent configuration stream ended before catalog update"),
            onSome: Effect.succeed,
          })),
        )
        const primary = changed.bySlot.primary
        expect(primary._tag).toBe("Ready")
        if (primary._tag === "Ready") {
          expect(primary.config.profile).toEqual({
            contextWindow: 16_384,
            maxOutputTokens: 2_048,
          })
        }
      }).pipe(Effect.provide(harness.layer))
    })))
  })
})
