import { createElement } from "react"
import { act, create, type ReactTestRenderer } from "react-test-renderer"
import { RegistryContext } from "@effect-atom/atom-react"
import * as Registry from "@effect-atom/atom/Registry"
import { Effect, Layer } from "effect"
import { Client as EffectQueryClient } from "@magnitudedev/effect-query"
import { describe, expect, it } from "vitest"
import {
  AgentClientProvider,
  useLocalModelActions,
  useLocalModelsSelector,
  type AgentClient,
} from "../index"
import { MagnitudeBoundary, type Change, type LocalModelsState } from "@magnitudedev/sdk"
import { Queue, Stream } from "effect"
import { clientServicesLayer, type ClientServices } from "../state/client-services"
import type { AcnClientRequirements } from "../state/agent-client"
import { fakeAcnImplementationsLayer } from "../state/fake-acn-implementations"

const localModelsState: LocalModelsState = {
  inventoryState: { _tag: "Ready" },
  models: [],
  discoveryState: { _tag: "Ready", progress: [] },
}

;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true

const makeFakeAgentClient = (
  onGetLocalModels: () => void,
  options?: {
    readonly getLocalModels?: () => Effect.Effect<unknown, unknown>
  },
) => {
  const request = (tag: string): Effect.Effect<unknown, unknown> => {
    if (tag === "GetLocalModels") {
      onGetLocalModels()
      return options?.getLocalModels?.() ?? Effect.succeed({ state: localModelsState })
    }
    return Effect.dieMessage(`Unexpected RPC in local-model lifecycle test: ${tag}`)
  }
  const changes = Effect.runSync(Queue.unbounded<Change>())
  const implementations = fakeAcnImplementationsLayer(
    request,
    (tag) => tag === "StreamChanges" ? Stream.fromQueue(changes) : Stream.never,
  )
  const client: AgentClient = EffectQueryClient.make<typeof MagnitudeBoundary, AcnClientRequirements, never, ClientServices, never>(
    MagnitudeBoundary,
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
  it("does not refetch GetLocalModels when switching menu consumers", async () => {
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

  it("keeps invalidation subscribed after the initial GetLocalModels request fails", async () => {
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
      await Effect.runPromise(invalidations.publish({ query: "GetModelSlots", revision: 1 }))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(calls).toBe(callsBeforeInvalidation)

    await act(async () => {
      await Effect.runPromise(invalidations.publish({ query: "GetLocalModels", revision: 2 }))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(calls).toBe(callsBeforeInvalidation + 1)

    await act(async () => renderer.unmount())
    registry.dispose()
  })
})
