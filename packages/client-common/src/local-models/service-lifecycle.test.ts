import { createElement } from "react"
import { act, create, type ReactTestRenderer } from "react-test-renderer"
import { Atom, RegistryContext } from "@effect-atom/atom-react"
import * as Registry from "@effect-atom/atom/Registry"
import { Context, Deferred, Effect, Layer, PubSub, Stream } from "effect"
import { Client as EffectQueryClient } from "@magnitudedev/effect-query"
import { describe, expect, it } from "vitest"
import {
  AgentClientProvider,
  useLocalModelActions,
  useLocalModelsSelector,
  type AgentClientInstance,
} from "../index"
import { AcnRpcClientTag, type AcnRpcClient, type LocalModelsState } from "@magnitudedev/sdk"
import { clientServicesLayer, type ClientServices } from "../state/client-services"

const localModelsState: LocalModelsState = {
  inventoryState: { _tag: "Ready" },
  models: [],
  discoveryState: { _tag: "Ready", progress: [] },
}

;(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean })
  .IS_REACT_ACT_ENVIRONMENT = true

interface FakeRpcClient {
  (tag: string): unknown
}

let nextFakeAgentClientId = 0

const makeFakeAgentClient = (
  onGetLocalModels: () => void,
  onWatchMirroredStates: () => void,
  options?: {
    readonly getLocalModels?: () => Effect.Effect<unknown, unknown>
    readonly watchMirroredStates?: () => Stream.Stream<unknown, unknown>
  },
): AgentClientInstance => {
  const FakeAgentClient = Context.GenericTag<FakeRpcClient>(
    `FakeAgentClient-${nextFakeAgentClientId++}`,
  )
  const service: FakeRpcClient = (tag) => {
    if (tag === "GetLocalModels") {
      onGetLocalModels()
      return options?.getLocalModels?.() ?? Effect.succeed({ state: localModelsState })
    }
    if (tag === "WatchMirroredStates") {
      onWatchMirroredStates()
      return options?.watchMirroredStates?.() ?? Stream.never
    }
    return Effect.dieMessage(`Unexpected RPC in local-model lifecycle test: ${tag}`)
  }
  const layer = Layer.succeed(FakeAgentClient, service)
  const runtime = Atom.runtime(layer)
  const effectQuery = EffectQueryClient.make<AcnRpcClientTag, never, ClientServices, never>(
    Layer.succeed(AcnRpcClientTag, service as unknown as AcnRpcClient),
    (client) => clientServicesLayer(client),
  )
  const mutation = () => Atom.fn(() => Effect.void)
  const tag = Object.assign(FakeAgentClient, {
    layer,
    runtime,
    mutation,
    effectQuery,
  })
  return { rpc: tag, effectQuery } as unknown as AgentClientInstance
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
    let watches = 0
    const client = makeFakeAgentClient(() => calls++, () => watches++)
    const registry = Registry.make({ defaultIdleTTL: 5_000 })
    let renderer!: ReactTestRenderer

    await act(async () => {
      renderer = create(renderHarness(registry, client, "models"))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(watches).toBe(1)
    expect(calls).toBe(1)

    await act(async () => {
      renderer.update(renderHarness(registry, client, "catalog"))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(watches).toBe(1)
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
    let watches = 0
    const watchConnected = Effect.runSync(Deferred.make<void>())
    const invalidations = Effect.runSync(PubSub.unbounded<{
      readonly _tag: "changed"
      readonly id: string
      readonly revision: number
    }>())
    const client = makeFakeAgentClient(
      () => calls++,
      () => watches++,
      {
        getLocalModels: () => calls === 1
          ? Effect.fail("temporarily unavailable")
          : Effect.succeed({ state: localModelsState }),
        watchMirroredStates: () => Stream.unwrap(Deferred.await(watchConnected).pipe(
          Effect.as(Stream.fromPubSub(invalidations)),
        )),
      },
    )
    const registry = Registry.make({ defaultIdleTTL: 5_000 })
    let renderer!: ReactTestRenderer

    await act(async () => {
      renderer = create(renderHarness(registry, client, "models"))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(watches).toBe(1)
    expect(calls).toBe(1)

    await act(async () => {
      await Effect.runPromise(Deferred.succeed(watchConnected, undefined))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    const callsBeforeInvalidation = calls

    await act(async () => {
      await Effect.runPromise(PubSub.publish(invalidations, {
        _tag: "changed",
        id: "GetModelSlots",
        revision: 1,
      } as const))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(calls).toBe(callsBeforeInvalidation)

    await act(async () => {
      await Effect.runPromise(PubSub.publish(invalidations, {
        _tag: "changed",
        id: "GetLocalModels",
        revision: 2,
      } as const))
      await Effect.runPromise(Effect.sleep("10 millis"))
    })
    expect(calls).toBe(callsBeforeInvalidation + 1)

    await act(async () => renderer.unmount())
    registry.dispose()
  })
})
