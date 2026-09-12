import { createElement } from "react"
import { act, create, type ReactTestRenderer } from "react-test-renderer"
import { RegistryContext } from "@effect-atom/atom-react"
import * as Registry from "@effect-atom/atom/Registry"
import { Deferred, Effect, Layer, Option } from "effect"
import { Client as EffectQueryClient } from "@magnitudedev/effect-query"
import { describe, expect, it } from "vitest"
import {
  AgentClientProvider,
  useLocalModelActions,
  useLocalModelMutations,
  useLocalModelStopStatus,
  useLocalModelsSelector,
  type AgentClient,
} from "../index"
import { CatalogFormModelIdSchema, type Change, type LocalModelsState } from "@magnitudedev/sdk"
import { Queue, Stream } from "effect"
import { clientServicesLayer, type ClientServices } from "../state/client-services"
import type { AcnClientRequirements } from "../state/agent-client"
import { fakeAcnImplementationsLayer } from "../state/fake-acn-implementations"
import { MagnitudeOperations } from "../state/application-operations"

const localModelsState: LocalModelsState = {
  preparation: {
    discovery: { complete: true, modelsFound: 0 },
    assessment: { complete: true, settledModels: 0, totalModels: 0 },
  },
  models: [],
}

;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true

const makeFakeAgentClient = (
  onGetModelCatalog: () => void,
  options?: {
    readonly getLocalModels?: () => Effect.Effect<unknown, unknown>
    readonly command?: (tag: string, payload: unknown) => Effect.Effect<unknown, unknown>
  },
) => {
  const request = (tag: string, payload: unknown): Effect.Effect<unknown, unknown> => {
    if (tag === "GetModelCatalog") {
      onGetModelCatalog()
      return options?.getLocalModels?.() ?? Effect.succeed({
        _tag: "Ready",
        providers: [],
        models: localModelsState.models.map((product) => ({
          _tag: "Local",
          product,
          offering: { _tag: "None" },
        })),
        failures: [],
        localModelPreparation: localModelsState.preparation,
      })
    }
    if (options?.command) return options.command(tag, payload)
    return Effect.dieMessage(`Unexpected RPC in local-model lifecycle test: ${tag}`)
  }
  const changes = Effect.runSync(Queue.unbounded<Change>())
  const implementations = fakeAcnImplementationsLayer(
    request,
    (tag) => tag === "StreamChanges" ? Stream.fromQueue(changes) : Stream.never,
  )
  const client: AgentClient = EffectQueryClient.make<typeof MagnitudeOperations, AcnClientRequirements, never, ClientServices, never>(
    MagnitudeOperations,
    implementations,
    (effectQuery) => clientServicesLayer(effectQuery),
  )
  const invalidations = {
    publish: (change: Change) => Queue.offer(changes, change).pipe(Effect.asVoid),
  }
  return { client, invalidations }
}

const selectModels = (state: LocalModelsState) => state.models
const sameModels = (left: LocalModelsState["models"], right: LocalModelsState["models"]) => left === right

const ModelsProbe = () => {
  useLocalModelsSelector(selectModels, sameModels)
  useLocalModelActions()
  return null
}

const CatalogProbe = () => {
  useLocalModelsSelector(selectModels, sameModels)
  useLocalModelActions()
  return null
}

const Harness = ({ root }: { readonly root: "models" | "catalog" }) =>
  createElement(root === "models" ? ModelsProbe : CatalogProbe)

const renderHarness = (
  registry: Registry.Registry,
  client: AgentClient,
  root: "models" | "catalog",
) => createElement(
  RegistryContext.Provider,
  { value: registry },
  createElement(
    AgentClientProvider,
    { tag: client, children: createElement(Harness, { root }) },
  ),
)

describe("local model query lifecycle", () => {
  it("does not refetch GetModelCatalog when switching menu consumers", async () => {
    let calls = 0
    const { client } = makeFakeAgentClient(() => calls++)
    const registry = Registry.make({ defaultIdleTTL: 5_000 })
    let renderer!: ReactTestRenderer

    await act(async () => {
      renderer = create(renderHarness(registry, client, "models"))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(calls).toBe(1)

    await act(async () => {
      renderer.update(renderHarness(registry, client, "catalog"))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(calls).toBe(1)

    await act(async () => {
      renderer.update(renderHarness(registry, client, "models"))
      await Effect.runPromise(Effect.sleep("10 millis"))
      renderer.unmount()
    })
    expect(calls).toBe(1)
    registry.dispose()
  })

  it("keeps invalidation subscribed after the initial GetModelCatalog request fails", async () => {
    let calls = 0
    const { client, invalidations } = makeFakeAgentClient(
      () => calls++,
      {
        getLocalModels: () => calls === 1
          ? Effect.fail("temporarily unavailable")
          : Effect.succeed({ state: localModelsState }),
      },
    )
    const registry = Registry.make({ defaultIdleTTL: 5_000 })
    let renderer!: ReactTestRenderer

    await act(async () => {
      renderer = create(renderHarness(registry, client, "models"))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(calls).toBe(1)
    const callsBeforeInvalidation = calls

    await act(async () => {
      await Effect.runPromise(invalidations.publish({ operation: "GetModelSlots" }))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(calls).toBe(callsBeforeInvalidation)

    await act(async () => {
      await Effect.runPromise(invalidations.publish({ operation: "GetModelCatalog" }))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(calls).toBe(callsBeforeInvalidation + 1)

    await act(async () => renderer.unmount())
    registry.dispose()
  })
})


describe("shared local-model command hooks", () => {
  it("keeps different model commands concurrent and shares Stop failure across consumers", async () => {
    const release = Effect.runSync(Deferred.make<void>())
    const started: unknown[] = []
    let completed = 0
    const { client } = makeFakeAgentClient(() => {}, { command: (tag, payload) => {
      if (tag === "SyncLocalModel") return Effect.sync(() => started.push(payload)).pipe(
        Effect.zipRight(Deferred.await(release)), Effect.tap(() => Effect.sync(() => completed++)),
      )
      if (tag === "StopActiveLocalModel") return Effect.fail({ message: "Cleanup is incomplete. Try Stop again." })
      return Effect.dieMessage(`Unexpected command: ${tag}`)
    } })
    const registry = Registry.make({ defaultIdleTTL: 5_000 })
    let actions!: ReturnType<typeof useLocalModelMutations>
    const statuses = new Map<number, ReturnType<typeof useLocalModelStopStatus>>()
    const Probe = ({ id }: { id: number }) => {
      const currentActions = useLocalModelMutations()
      const status = useLocalModelStopStatus()
      if (id === 0) actions = currentActions
      statuses.set(id, status)
      return null
    }
    let renderer!: ReactTestRenderer
    try {
      await act(async () => {
        renderer = create(createElement(RegistryContext.Provider, { value: registry },
          createElement(AgentClientProvider, { tag: client, children: [0, 1].map(id => createElement(Probe, { key: id, id })) })))
        await Effect.runPromise(Effect.sleep("20 millis"))
      })
      const first = CatalogFormModelIdSchema.make("first:gguf:q4")
      const second = CatalogFormModelIdSchema.make("second:gguf:q4")
      await act(async () => {
        actions.install(first)
        actions.install(second)
        await Effect.runPromise(Effect.sleep("20 millis"))
      })
      expect(started).toEqual([{ modelId: first }, { modelId: second }])
      expect(completed).toBe(0)
      await act(async () => {
        await Effect.runPromise(Deferred.succeed(release, undefined))
        await Effect.runPromise(Effect.sleep("20 millis"))
      })
      expect(completed).toBe(2)
      await act(async () => {
        actions.stop()
        await Effect.runPromise(Effect.sleep("20 millis"))
      })
      for (const id of [0, 1]) expect(statuses.get(id)).toEqual({ pending: false, failure: Option.some("Cleanup is incomplete. Try Stop again.") })
    } finally {
      if (renderer) await act(async () => renderer.unmount())
      registry.dispose()
    }
  })
})
