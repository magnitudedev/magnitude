import { createElement } from "react"
import { act, create, type ReactTestRenderer } from "react-test-renderer"
import { Atom, RegistryContext } from "@effect-atom/atom-react"
import * as Registry from "@effect-atom/atom/Registry"
import { Context, Effect, Layer } from "effect"
import { Client as EffectQueryClient } from "@magnitudedev/effect-query"
import { describe, expect, it } from "vitest"
import {
  AgentClientProvider,
  useLocalModelActions,
  useLocalModelsSelector,
  type AgentClientInstance,
} from "../index"
import { Acn, type AcnTransport, type Change, type LocalModelsState } from "@magnitudedev/sdk"
import { Queue, Stream } from "effect"
import { clientServicesLayer, type ClientServices } from "../state/client-services"
import { fakeAcnTransport } from "../state/fake-acn-transport"

const localModelsState: LocalModelsState = {
  inventoryState: { _tag: "Ready" },
  models: [],
  discoveryState: { _tag: "Ready", progress: [] },
}

;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true

interface FakeRpcClient {
  (tag: string): Effect.Effect<unknown, unknown>
}

let nextFakeAgentClientId = 0

const makeFakeAgentClient = (
  onGetLocalModels: () => void,
  options?: {
    readonly getLocalModels?: () => Effect.Effect<unknown, unknown>
  },
) => {
  const FakeAgentClient = Context.GenericTag<FakeRpcClient>(
    `FakeAgentClient-${nextFakeAgentClientId++}`,
  )
  const service: FakeRpcClient = (tag) => {
    if (tag === "GetLocalModels") {
      onGetLocalModels()
      return options?.getLocalModels?.() ?? Effect.succeed({ state: localModelsState })
    }
    return Effect.dieMessage(`Unexpected RPC in local-model lifecycle test: ${tag}`)
  }
  const layer = Layer.succeed(FakeAgentClient, service)
  const runtime = Atom.runtime(layer)
  const changes = Effect.runSync(Queue.unbounded<Change>())
  const transport = fakeAcnTransport(
    (tag) => service(tag),
    (tag) => tag === "StreamChanges" ? Stream.fromQueue(changes) : Stream.never,
  )
  const effectQuery = EffectQueryClient.make<AcnTransport, never, ClientServices, never>(
    Layer.succeed(Acn.Client, transport),
    (client) => clientServicesLayer(client),
  )
  const invalidations = {
    publish: (change: Change) => Queue.offer(changes, change).pipe(Effect.asVoid),
  }
  const mutation = () => Atom.fn(() => Effect.void)
  const tag = Object.assign(FakeAgentClient, {
    layer,
    runtime,
    mutation,
    effectQuery,
  })
  return {
    client: { rpc: tag, effectQuery } as unknown as AgentClientInstance,
    invalidations,
  }
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
  client: AgentClientInstance,
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
