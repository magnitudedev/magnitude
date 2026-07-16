import { useMemo } from "react"
import { Cause, Effect, Stream } from "effect"
import { Atom, Result, useAtomMount, useAtomValue } from "@effect-atom/atom-react"
import * as Reactivity from "@effect/experimental/Reactivity"
import { useAgentClient } from "../state/agent-client-context"
import type { ModelCatalog, ModelSlots } from "@magnitudedev/sdk"
import { makeResidentWatchRegistry } from "../state/reactive-rpc-watch-registry"

export type ReactiveRpcResource = "modelCatalog" | "modelSlots"
export type ReactiveRpcConfig =
  | {
    readonly query: readonly ["GetModelCatalog", Record<string, never>]
    readonly watch: readonly ["WatchModelCatalog", Record<string, never>]
    readonly reactivityKey: "modelCatalog"
  }
  | {
    readonly query: readonly ["GetModelSlots", Record<string, never>]
    readonly watch: readonly ["WatchModelSlots", Record<string, never>]
    readonly reactivityKey: "modelSlots"
  }

const residentWatches = makeResidentWatchRegistry<ReactiveRpcResource>()

/**
 * A query-backed server resource with a resident invalidation subscription.
 * The stream never becomes a second state store: it only reruns the query.
 */
interface ReactiveRpcValues {
  readonly modelCatalog: ModelCatalog
  readonly modelSlots: ModelSlots
}

export function useReactiveRpc<Config extends ReactiveRpcConfig>(
  config: Config,
): Result.Result<ReactiveRpcValues[Config["reactivityKey"]], unknown> {
  const client = useAgentClient()
  const resource = config.reactivityKey
  const queryAtom = useMemo(() => resource === "modelCatalog"
    ? client.query("GetModelCatalog", {}, { reactivityKeys: [resource] })
    : client.query("GetModelSlots", {}, { reactivityKeys: [resource] }), [client, resource])
  const watchAtom = useMemo(() => residentWatches.getOrCreate(client, resource, () => client.runtime.atom(Effect.gen(function* () {
    const rpc = yield* client
    const stream = resource === "modelCatalog"
      ? rpc("WatchModelCatalog", {})
      : rpc("WatchModelSlots", {})
    yield* stream.pipe(
      Stream.filter((event) => event._tag === "changed"),
      Stream.tap(() => Reactivity.invalidate([resource])),
      Stream.runDrain,
    )
  }).pipe(
    Effect.catchAllCause((cause) => Cause.isInterruptedOnly(cause)
      ? Effect.void
      : Effect.logError(`[${resource}] ${Cause.pretty(cause)}`)),
  ))), [client, resource])

  useAtomMount(watchAtom)
  return useAtomValue(queryAtom as unknown as Atom.Atom<Result.Result<ReactiveRpcValues[Config["reactivityKey"]], unknown>>)
}

export const useModelCatalog = () => useReactiveRpc({
  query: ["GetModelCatalog", {}],
  watch: ["WatchModelCatalog", {}],
  reactivityKey: "modelCatalog",
})

export const useModelSlots = () => useReactiveRpc({
  query: ["GetModelSlots", {}],
  watch: ["WatchModelSlots", {}],
  reactivityKey: "modelSlots",
})
