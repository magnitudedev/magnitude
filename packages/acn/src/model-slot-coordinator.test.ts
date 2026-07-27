import { describe, expect, it } from "vitest"
import {
  Deferred,
  Effect,
  Fiber,
  Layer,
  Option,
  Ref,
  Stream,
  SubscriptionRef,
} from "effect"
import {
  LocalModelMutationFailed,
  ModelOfferingTargetIdSchema,
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  PRIMARY_SLOT_ID,
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
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { AcnActivityTrackerLive } from "./activity-tracker"
import { AcnShutdownLive } from "./acn-shutdown"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelRuntime } from "./local-model-runtime"
import { LocalProviderOfferings } from "./local-provider-offerings"
import { MirroredStateChangesLive } from "./mirrored-state"
import { ModelConfiguration } from "./model-configuration"
import { ModelSlotCoordinator, ModelSlotCoordinatorLive } from "./model-slot-coordinator"
import { ProviderModelCatalog } from "./provider-model-catalog"

const providerModelId = ProviderModelIdSchema.make("local:test")
const effort = ReasoningEffortSchema.make("none")
const selection: SlotSelection = {
  providerId: LOCAL_PROVIDER_ID,
  providerModelId,
  reasoningEffort: effort,
}
const packageId = ModelPackageIdSchema.make("test-package")
const fallbackProviderModelId = ProviderModelIdSchema.make("local:fallback")

const catalogModel: ProviderModelCatalogEntry = {
  providerId: LOCAL_PROVIDER_ID,
  providerModelId,
  modelFamilyId: Option.none(),
  displayName: "Test local model",
  supportedSlots: [PRIMARY_SLOT_ID],
  contextWindow: 8_192,
  maxOutputTokens: 1_024,
  memory: Option.none(),
  capabilities: {
    vision: false,
    tools: true,
    structuredOutput: true,
    reasoning: {
      supported: false,
      efforts: [],
      defaultEffort: Option.none(),
    },
  },
  availability: { _tag: "Available" },
  pricing: Option.none(),
}

const offering = {
  providerModelId,
  modelId: ModelOfferingTargetIdSchema.make("test-model"),
  configuration: {
    id: ModelServingConfigurationIdSchema.make("test-configuration"),
    target: {
      _tag: "Package",
      package: { id: packageId },
    },
  },
  origin: { _tag: "UserConfigured" },
  capabilities: catalogModel.capabilities,
} as LocalProviderOffering

const fallbackCatalogModel: ProviderModelCatalogEntry = {
  ...catalogModel,
  providerModelId: fallbackProviderModelId,
  displayName: "Previously installed local model",
}

const unavailableCatalogModel: ProviderModelCatalogEntry = {
  ...catalogModel,
  availability: { _tag: "Disabled", reason: "model_unavailable" },
}

const makeHarnessWith = (options: {
  readonly downloadPending?: boolean
  readonly availableFallback?: boolean
  readonly unassigned?: boolean
} = {}) => Effect.gen(function* () {
  const configured = yield* SubscriptionRef.make({
    slots: {
      primary: options.unassigned ? Option.none<SlotSelection>() : Option.some(selection),
      secondary: Option.none<SlotSelection>(),
    },
    localModelRecency: {
      primary: options.availableFallback
        ? [fallbackProviderModelId, providerModelId]
        : [providerModelId],
      secondary: [],
    },
    favoriteModels: [],
    localProviderOfferings: [],
    dismissedDownloadFailures: [],
    contextLimits: resolveContextLimitPolicy({}),
  })
  const resident = yield* Ref.make(false)
  const loadCalls = yield* Ref.make(0)
  const unloadCalls = yield* Ref.make(0)
  const loadGate = yield* Ref.make<Option.Option<Deferred.Deferred<void>>>(Option.none())
  const unloadGate = yield* Ref.make<Option.Option<Deferred.Deferred<void>>>(Option.none())
  const residencyReadGate = yield* Ref.make<Option.Option<{
    readonly entered: Deferred.Deferred<void>
    readonly release: Deferred.Deferred<void>
  }>>(Option.none())
  const loadFailure = yield* Ref.make<Option.Option<LocalModelMutationFailed>>(Option.none())
  const loadDefect = yield* Ref.make(false)
  const unloadEntered = yield* Deferred.make<void>()

  const runtime = LocalModelRuntime.of({
    load: () => Effect.gen(function* () {
      yield* Ref.update(loadCalls, (count) => count + 1)
      const gate = yield* Ref.get(loadGate)
      if (Option.isSome(gate)) yield* Deferred.await(gate.value)
      if (yield* Ref.get(loadDefect)) return yield* Effect.die("load defect")
      const failure = yield* Ref.get(loadFailure)
      if (Option.isSome(failure)) return yield* failure.value
      yield* Ref.set(resident, true)
    }),
    unload: () => Effect.gen(function* () {
      yield* Ref.update(unloadCalls, (count) => count + 1)
      yield* Deferred.succeed(unloadEntered, undefined)
      const gate = yield* Ref.get(unloadGate)
      if (Option.isSome(gate)) yield* Deferred.await(gate.value)
      yield* Ref.set(resident, false)
    }),
    isResident: () => Effect.gen(function* () {
      const gate = yield* Ref.get(residencyReadGate)
      if (Option.isSome(gate)) {
        yield* Deferred.succeed(gate.value.entered, undefined)
        yield* Deferred.await(gate.value.release)
      }
      return yield* Ref.get(resident)
    }),
    changes: Stream.empty,
  })

  const dependencies = Layer.mergeAll(
    Layer.succeed(ModelConfiguration, ModelConfiguration.of({
      get: SubscriptionRef.get(configured),
      changes: configured.changes,
      updateSlot: (slotId, next) => SubscriptionRef.update(configured, (current) => ({
        ...current,
        slots: { ...current.slots, [slotId]: next },
      })),
      recordUse: () => Effect.void,
      setFavorite: () => Effect.void,
    })),
    Layer.succeed(LocalModelPackages, LocalModelPackages.of({
      snapshot: Effect.succeed({ revision: 0, state: { entries: [] } }),
      changes: Stream.empty,
      installedPackageIds: Effect.succeed(options.downloadPending ? new Set() : new Set([packageId])),
      downloadTarget: () => Effect.void,
      cancelTargetDownload: () => Effect.void,
      dismissTargetFailure: () => Effect.void,
      removeTargetPackages: () => Effect.void,
      refresh: Effect.void,
    })),
    Layer.succeed(LocalModelRuntime, runtime),
    Layer.succeed(LocalProviderOfferings, LocalProviderOfferings.of({
      list: Effect.succeed([offering]),
      changes: Stream.empty,
      resolve: () => Effect.succeed(offering),
      save: () => Effect.succeed(offering),
    })),
    Layer.succeed(ProviderModelCatalog, ProviderModelCatalog.of({
      snapshot: Effect.succeed({
        revision: 0,
        state: new ProviderModelCatalogReady({
          providers: [{
            providerId: ProviderIdSchema.make("local"),
            displayName: "Local",
            authentication: "NotRequired",
            availability: { _tag: "Available" },
          }],
          models: options.downloadPending
            ? options.availableFallback
              ? [unavailableCatalogModel, fallbackCatalogModel]
              : []
            : [catalogModel],
        }),
      }),
      changes: Stream.empty,
      refresh: () => Effect.void,
    })),
    MirroredStateChangesLive,
    AcnActivityTrackerLive("30 minutes").pipe(Layer.provide(AcnShutdownLive)),
  )

  return {
    layer: ModelSlotCoordinatorLive.pipe(Layer.provide(dependencies)),
    controls: {
      resident,
      loadCalls,
      unloadCalls,
      loadGate,
      unloadGate,
      residencyReadGate,
      loadFailure,
      loadDefect,
      unloadEntered,
    },
  }
})

const makeHarness = makeHarnessWith()

describe("ModelSlotCoordinator owned transitions", () => {
  it("accepts a saved local offering while its packages are still downloading", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarnessWith({ downloadPending: true, unassigned: true })
      yield* Effect.gen(function* () {
        const coordinator = yield* ModelSlotCoordinator
        yield* coordinator.updateModelSlot(PRIMARY_SLOT_ID, Option.some(selection))
        const slot = (yield* coordinator.snapshot).state.slots.primary
        expect(slot._tag).toBe("Blocked")
        if (slot._tag === "Blocked") {
          expect(slot.reason._tag).toBe("ModelUnavailable")
          expect(slot.selection).toEqual(selection)
        }
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("does not recover an unavailable offered selection to an older installed model", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarnessWith({
        downloadPending: true,
        availableFallback: true,
      })
      yield* Effect.gen(function* () {
        const coordinator = yield* ModelSlotCoordinator
        const slot = (yield* coordinator.snapshot).state.slots.primary
        expect(slot._tag).toBe("Blocked")
        if (slot._tag === "Blocked") {
          expect(slot.selection.providerModelId).toBe(providerModelId)
          expect(slot.reason._tag).toBe("ModelUnavailable")
        }
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("keeps one load owner when an equivalent waiter is interrupted", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness
      const loadRelease = yield* Deferred.make<void>()
      yield* Ref.set(harness.controls.loadGate, Option.some(loadRelease))

      yield* Effect.gen(function* () {
        const coordinator = yield* ModelSlotCoordinator
        yield* coordinator.loadModel(PRIMARY_SLOT_ID)
        yield* coordinator.loadModel(PRIMARY_SLOT_ID)
        yield* Effect.yieldNow()
        expect((yield* coordinator.snapshot).state.slots.primary._tag).toBe("LoadingLocalModel")
        expect(yield* Ref.get(harness.controls.loadCalls)).toBe(1)
        expect((yield* Effect.either(coordinator.unloadModel(PRIMARY_SLOT_ID)))._tag).toBe("Left")

        const waiter = yield* Effect.scoped(
          coordinator.acquireLocalModel(PRIMARY_SLOT_ID, providerModelId),
        ).pipe(Effect.fork)
        yield* Fiber.interrupt(waiter)
        expect(yield* Ref.get(harness.controls.loadCalls)).toBe(1)

        const joiningWaiter = yield* Effect.scoped(
          coordinator.acquireLocalModel(PRIMARY_SLOT_ID, providerModelId),
        ).pipe(Effect.fork)
        yield* Deferred.succeed(loadRelease, undefined)
        yield* Fiber.join(joiningWaiter)
        expect((yield* coordinator.snapshot).state.slots.primary._tag).toBe("Ready")
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("holds the transition lock across satisfaction and admission decisions", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness
      yield* Effect.gen(function* () {
        const coordinator = yield* ModelSlotCoordinator
        yield* coordinator.loadModel(PRIMARY_SLOT_ID)
        yield* Effect.scoped(coordinator.acquireLocalModel(PRIMARY_SLOT_ID, providerModelId))

        const readEntered = yield* Deferred.make<void>()
        const releaseRead = yield* Deferred.make<void>()
        const releaseUnload = yield* Deferred.make<void>()
        yield* Ref.set(harness.controls.residencyReadGate, Option.some({
          entered: readEntered,
          release: releaseRead,
        }))
        yield* Ref.set(harness.controls.unloadGate, Option.some(releaseUnload))

        const satisfiedLoad = yield* coordinator.loadModel(PRIMARY_SLOT_ID).pipe(Effect.fork)
        yield* Deferred.await(readEntered)
        const unload = yield* coordinator.unloadModel(PRIMARY_SLOT_ID).pipe(Effect.fork)
        yield* Effect.yieldNow()
        expect(Option.isNone(yield* Deferred.poll(harness.controls.unloadEntered))).toBe(true)

        yield* Deferred.succeed(releaseRead, undefined)
        yield* Fiber.join(satisfiedLoad)
        yield* Deferred.await(harness.controls.unloadEntered)
        expect(yield* Ref.get(harness.controls.unloadCalls)).toBe(1)
        yield* Deferred.succeed(releaseUnload, undefined)
        yield* Fiber.join(unload)
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("does not admit a caller canceled while waiting for the transition lock", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness
      yield* Effect.gen(function* () {
        const coordinator = yield* ModelSlotCoordinator
        yield* coordinator.loadModel(PRIMARY_SLOT_ID)
        yield* Effect.scoped(coordinator.acquireLocalModel(PRIMARY_SLOT_ID, providerModelId))

        const readEntered = yield* Deferred.make<void>()
        const releaseRead = yield* Deferred.make<void>()
        yield* Ref.set(harness.controls.residencyReadGate, Option.some({
          entered: readEntered,
          release: releaseRead,
        }))

        const satisfiedLoad = yield* coordinator.loadModel(PRIMARY_SLOT_ID).pipe(Effect.fork)
        yield* Deferred.await(readEntered)
        const canceledUnload = yield* coordinator.unloadModel(PRIMARY_SLOT_ID).pipe(Effect.fork)
        yield* Fiber.interrupt(canceledUnload)
        yield* Deferred.succeed(releaseRead, undefined)
        yield* Fiber.join(satisfiedLoad)
        yield* Effect.yieldNow()

        expect(yield* Ref.get(harness.controls.unloadCalls)).toBe(0)
        expect((yield* coordinator.snapshot).state.slots.primary._tag).toBe("Ready")
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("terminalizes a failed owner as a blocked slot", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness
      yield* Ref.set(harness.controls.loadFailure, Option.some(new LocalModelMutationFailed({
        code: "test_load_failure",
        message: "load failed",
        retryable: true,
      })))
      yield* Effect.gen(function* () {
        const coordinator = yield* ModelSlotCoordinator
        yield* coordinator.loadModel(PRIMARY_SLOT_ID)
        const result = yield* Effect.either(Effect.scoped(
          coordinator.acquireLocalModel(PRIMARY_SLOT_ID, providerModelId),
        ))
        expect(result._tag).toBe("Left")
        const slot = (yield* coordinator.snapshot).state.slots.primary
        expect(slot._tag).toBe("Blocked")
        if (slot._tag === "Blocked") {
          expect(slot.reason._tag).toBe("LocalModelLoadFailed")
        }
      }).pipe(Effect.provide(harness.layer))
    })))
  })

  it("terminalizes an owner defect instead of stranding loading state", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const harness = yield* makeHarness
      yield* Ref.set(harness.controls.loadDefect, true)
      yield* Effect.gen(function* () {
        const coordinator = yield* ModelSlotCoordinator
        yield* coordinator.loadModel(PRIMARY_SLOT_ID)
        const result = yield* Effect.exit(Effect.scoped(
          coordinator.acquireLocalModel(PRIMARY_SLOT_ID, providerModelId),
        ))
        expect(result._tag).toBe("Failure")
        expect((yield* coordinator.snapshot).state.slots.primary._tag).toBe("Blocked")
      }).pipe(Effect.provide(harness.layer))
    })))
  })
})
