# `@magnitudedev/effect-query`

Effect-native query and mutation state built on Effect Atom. Definitions retain their exact input,
data, expected-error, and service-requirement types.

## Relationship to TanStack Query

This package intentionally preserves [TanStack Query](https://tanstack.com/query/latest)'s
established query and mutation concepts so that cache behavior and API responsibilities remain
familiar. It expresses those concepts with Effect and Effect Atom rather than React hooks,
Promises, or an untyped error channel. The goal is
semantic parity, not a line-for-line port: exact Effect error and requirement types, atom identity,
and Effect services remain native to this package.

| TanStack Query concept | `effect-query` equivalent | Effect-native difference |
| --- | --- | --- |
| Query options (`queryKey`, `queryFn`) | `Query.make(name, { key, effect })` | The fetch operation is an `Effect`; its error and service requirements remain typed. |
| `useQuery` | Observe the query atom with the relevant Effect Atom integration | The query itself is a framework-independent atom. |
| `fetchQuery` | `QueryClient.fetch` | Returns fresh retained data, joins active work, or fetches missing/stale data; preserves the exact error type. |
| `ensureQueryData` | `QueryClient.ensure` | Returns retained data immediately and revalidates stale data. |
| `prefetchQuery` | `QueryClient.prefetch` | Failure remains in query state and is not returned in the Effect error channel. |
| `invalidateQueries` | `QueryClient.invalidate` | Accepts typed definition/key filters and optionally refetches. |
| `refetchQueries` | `QueryClient.refetch` | Heterogeneous failures are collected in `QueryBatchError`. |
| `cancelQueries` | `QueryClient.cancel` | Cancellation is represented by Effect interruption. |
| `removeQueries` | `QueryClient.remove` | Removes matching query entries from the atom-backed cache. |
| `getQueryState` | `QueryClient.getState` | Returns `Option<Query.State<...>>` in an `Effect`. |
| `setQueryData` | `QueryClient.setData` | The updater receives `Option<Data>`. |
| Query cache retention (`gcTime`) | Query `gcTime` | Retains the complete canonical entry—data, status, invalidation, and in-flight coordination—after its last observer unmounts. |
| Mutation options (`mutationFn`, `scope`) | `Mutation.make({ effect, scope })` | The command and optional synchronization operation are typed Effects. |
| `mutate` / `mutateAsync` | Write the mutation atom / `Mutation.execute` | Atom writes support reactive use; `execute` composes directly in Effect workflows. |
| Mutation key | Exact mutation-definition identity | The definition is the typed identity, avoiding a parallel string-key contract. |
| `MutationState` | `Mutation.State<M>` | Preserves the exact mutation input, output, and error types. |
| `useMutationState` | `Mutation.state({ filters, select })` | Returns a derived `Atom` containing matching mutation states. |
| `useIsMutating` | `Mutation.isMutating(filters)` | Returns an `Atom<number>`. |
| Mutation cache retention (`gcTime`) | Mutation `gcTime` | Each invocation's `Mutation.State` remains selectable until collection. |

## Queries

Definitions are static domain values. Create them once, independently of any connection or Atom
runtime. A `Client` materializes those definitions for one application connection and owns the
shared query cache and mutation history:

```ts
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import * as Data from "effect/Data"
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import * as Option from "effect/Option"
import { Client, Query, QueryClient } from "@magnitudedev/effect-query"

class UserNotFound extends Data.TaggedError("UserNotFound")<{
  readonly id: string
}> {}

const userQuery = Query.make("User", {
  key: ({ id }: { readonly id: string }) => Data.tuple(id),
  effect: ({ id }) => id === "missing"
    ? Effect.fail(new UserNotFound({ id }))
    : Effect.succeed({ id, name: "Ada" }),
  staleTime: "30 seconds",
  gcTime: "5 minutes"
})

const client = Client.make(Layer.empty)
const registry = AtomRegistry.make()
const user = client.query(userQuery, { id: "1" })
const unmount = registry.mount(user)
const state = registry.get(user)
const current = AtomResult.value(state.result)
unmount()
```

Equivalent keys return the same query atom and share cached data and in-flight work. Structured keys
must be Effect `Data` or another value implementing Effect `Equal`; plain objects and arrays are
rejected because they have reference identity.

Like TanStack Query, observer lifetime and cache lifetime are distinct. Unmounting the final
observer does not reset a query: its complete canonical entry remains available until `gcTime`
expires. Remounting during that interval synchronously reads retained state. Concurrent fetches
join the active fetch, while repeated invalidations during an active replacement fetch coalesce
instead of starting parallel requests.

The keyed family returns that canonical query atom directly. Consumers and the registry therefore
hold the same object that the family weakly indexes; it must not index a temporary wrapper around
the atom, because collecting that wrapper would split one query key into multiple entries.

`QueryClient` also provides exact operations for cache control:

```ts
const cacheProgram = Effect.gen(function*() {
  yield* QueryClient.ensure(userQuery, { id: "1" }) // retained value, revalidate if stale
  yield* QueryClient.prefetch(userQuery, { id: "1" }) // failure stays in query state
  yield* QueryClient.invalidate(userQuery.match())
  yield* QueryClient.invalidate(userQuery.match({ id: "1" }), { refetch: false })
  yield* QueryClient.cancel({ stale: true })
  yield* QueryClient.setData(userQuery, { id: "1" }, (current) => Option.match(current, {
    onNone: () => ({ id: "1", name: "Grace" }),
    onSome: (value) => ({ ...value, name: "Grace" })
  }))
  yield* QueryClient.remove(userQuery.match({ id: "1" }))
})
```

`fetch` and `ensure` preserve the query's exact error type. Broad filtered `refetch`
instead returns `QueryBatchError`, because matches may have unrelated error types.

## RPC-backed definitions

RPC is an Effect service required by static definitions. Provide that service once when creating
the Effect Query client; no query-specific RPC adapter is needed:

```ts
import { FetchHttpClient } from "@effect/platform"
import { RpcClient, RpcClientError, RpcGroup, RpcSerialization } from "@effect/rpc"
import { Context, Effect, Layer } from "effect"
import { Client, Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import { MagnitudeRpcs } from "./protocol.js"

class AcnRpc extends Context.Tag("AcnRpc")<
  AcnRpc,
  RpcClient.RpcClient.Flat<RpcGroup.Rpcs<typeof MagnitudeRpcs>, RpcClientError.RpcClientError>
>() {}

const AcnRpcLive = Layer.scoped(
  AcnRpc,
  RpcClient.make(MagnitudeRpcs, { flatten: true })
).pipe(
  Layer.provide(RpcClient.layerProtocolHttp({ url: "http://127.0.0.1:3030/rpc" }).pipe(
    Layer.provide(RpcSerialization.layerNdjson),
    Layer.provide(FetchHttpClient.layer)
  ))
)

const sessionQuery = Query.make("Session", {
  key: ({ sessionId }: { readonly sessionId: string }) => Data.tuple(sessionId),
  effect: (payload) => Effect.flatMap(AcnRpc, (client) =>
    client("GetSession", payload))
})

const effectQuery = Client.make(AcnRpcLive)
const sessionAtom = effectQuery.query(sessionQuery, { sessionId: "session-1" })
```

Commands use the same service. `Client.make` supplies `QueryClient` to synchronization Effects:

```ts
const deleteSession = Mutation.make("DeleteSession", {
  effect: (payload: { readonly sessionId: string }) =>
    Effect.flatMap(AcnRpc, (client) => client("DeleteSession", payload)),
  synchronize: (_output, payload) =>
    QueryClient.remove(sessionQuery.match(payload)),
  scope: ({ sessionId }) => Mutation.MutationScope(`session:${sessionId}`)
})

const deleteSessionAtom = effectQuery.mutation(deleteSession)
```

Payload, success, domain-error, middleware-error, and RPC transport-error types are inferred from
the RPC group. `effect-query` adds cache policy and synchronization without restating the wire
contract.

## HTTP API-backed definitions

The same pattern applies to `AtomHttpApi.Tag`. Its generated client is an Effect service; the query
definition supplies only domain identity and cache policy:

```ts
import * as AtomHttpApi from "@effect-atom/atom/AtomHttpApi"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import { IcnApi } from "./icn-api.js"

class IcnHttpClient {}

const IcnHttp = AtomHttpApi.Tag<IcnHttpClient>()("IcnHttpClient", {
  api: IcnApi,
  httpClient: FetchHttpClient.layer,
  baseUrl: "http://127.0.0.1:8080"
})

const hardwareQuery = Query.make("Hardware", {
  key: () => Data.tuple("hardware"),
  effect: () => Effect.flatMap(IcnHttp, (client) =>
    client.system.getHardware({})),
  staleTime: "10 seconds"
})

const modelDownloadQuery = Query.make("ModelDownload", {
  key: ({ attemptId }: { readonly attemptId: string }) => Data.tuple(attemptId),
  effect: ({ attemptId }) => Effect.flatMap(IcnHttp, (client) =>
    client.models.getModelDownload({ path: { attempt_id: attemptId } }))
})

const icnQuery = Client.make(IcnHttp.layer)
const hardwareAtom = icnQuery.query(hardwareQuery, undefined)
```

Endpoint success, declared API errors, HTTP client failures, and Schema parse failures remain in the
query's inferred error channel. A generated `HttpApiClient` service can be used identically; the
essential input to `Query.make` is its typed Effect, not a transport-specific adapter.

## Remote atom pattern

A remote atom is ordinary composition, not another cache primitive:

```text
snapshot RPC → Query atom ← invalidate ← watch Stream
```

Install the remote watch as scoped Effect infrastructure. Notifications carry identity and
announce that authoritative state may have changed, while the query remains the only owner of
snapshot data:

```ts
import * as Stream from "effect/Stream"

const synchronizeModels = Effect.gen(function*() {
  const client = yield* AcnRpc

  yield* client("WatchModelChanges", {}).pipe(
    Stream.runForEach((change) =>
      QueryClient.invalidate(modelQuery.match({ modelId: change.modelId }))
    )
  )
})

const ModelSynchronizationLive = Layer.scopedDiscard(synchronizeModels)
```

Mutation synchronization uses the same cache primitives. Fetch the exact query when the command
must remain pending until the authoritative snapshot has been read:

```ts
const renameModel = Mutation.make("RenameModel", {
  effect: (payload: {
    readonly modelId: string
    readonly name: string
  }) => Effect.flatMap(AcnRpc, (client) =>
    client("RenameModel", payload)),

  synchronize: (_receipt, payload) => QueryClient.invalidate(
    modelQuery.match({ modelId: payload.modelId })
  ).pipe(
    Effect.zipRight(QueryClient.fetch(modelQuery, { modelId: payload.modelId })),
    Effect.asVoid
  )
})
```

The resulting “remote atom” is therefore a `Query` plus an optional scoped notification Stream.
RPC and HTTP transports supply Effects and Streams but do not need query-specific adapters. If an
API requires a stronger consistency guarantee than “refetch the authoritative snapshot,” that
guarantee belongs in the query's RPC Effect rather than in the generic cache.

## Observation and selection

Query atoms are ordinary Effect atoms:

```ts
const unsubscribe = registry.subscribe(user, (state) => {
  console.log(state.fetchStatus, state.result)
})

const nameAtom = Query.select(user, (value) => value.name)
const optionalUser = Query.when(Option.fromNullable(user))
```

`select` returns a derived read-only atom and does not create another cache entry. Cache operations
continue to target the source query atom.

### React

React observes the same query and mutation atoms through `@effect-atom/atom-react`; no query-specific
hooks or provider are needed:

```tsx
import {
  RegistryProvider,
  Result,
  useAtomSet,
  useAtomValue
} from "@effect-atom/atom-react"
import { createRoot } from "react-dom/client"

function Session({ sessionId }: { readonly sessionId: string }) {
  const state = useAtomValue(effectQuery.query(sessionQuery, { sessionId }))
  const remove = useAtomSet(effectQuery.mutation(deleteSession), { mode: "promise" })

  if (Result.isInitial(state.result)) return <p>Loading…</p>
  if (Result.isFailure(state.result)) return <p>Unable to load session.</p>

  return (
    <section aria-busy={state.result.waiting}>
      <h1>{state.result.value.title}</h1>
      <button onClick={() => void remove({ sessionId })}>Delete</button>
    </section>
  )
}

createRoot(document.getElementById("root")!).render(
  <RegistryProvider defaultIdleTTL={5_000}>
    <Session sessionId="session-1" />
  </RegistryProvider>
)
```

`RegistryProvider` owns the registry used by observation, remote-client runtimes, and the
connection-scoped Effect Query client, so reads, commands, and synchronization operate on the same
cache.

## Mutations

Mutation command and synchronization Effects retain separate error and requirement types. The
connection-scoped client supplies both application services and the query client used during
synchronization:

```ts
import { Mutation } from "@magnitudedev/effect-query"

const renameUser = Mutation.make("RenameUser", {
  effect: ({ id, name }: { readonly id: string; readonly name: string }) =>
    Effect.succeed({ id, name }),
  synchronize: (_output, input) =>
    QueryClient.invalidate(userQuery.match({ id: input.id })),
  scope: ({ id }) => Mutation.MutationScope(`user:${id}`)
})

const renameUserAtom = client.mutation(renameUser)
const renameProgram = Mutation.execute(renameUserAtom, { id: "1", name: "Grace" }).pipe(
  Effect.provideService(AtomRegistry.AtomRegistry, registry)
)
```

Invocations with the same scope are serialized. A synchronization failure is reported as
`MutationSynchronizationError<Output, SynchronizationError>`, retaining both the accepted command
output and the exact visibility error. Reactive aggregate state is available through
`QueryClient.isFetching`, `QueryClient.isMutating`, and `QueryClient.mutationState`.

The atom equivalents of TanStack Query's `useMutationState` and `useIsMutating` preserve the
mutation's input, output, and error types and support exact semantic scope selection:

```ts
const userScope = Mutation.MutationScope("user:1")
const mutationStatesAtom = Mutation.state({
  filters: { mutation: renameUser, scope: userScope },
})
const namesAtom = Mutation.state({
  filters: { mutation: renameUser, status: "success" },
  select: ({ input }) => input.name,
})
const isMutatingAtom = Mutation.isMutating({ mutation: renameUser, scope: userScope })
```

Derive the latest invocation state with `mutationStates.at(-1)` and a pending boolean with
`isMutating > 0`; these are consumer views rather than cache APIs. Resource lifecycle continues to
come from its query.

### Optimistic presentation from mutation state

For updates whose immediate presentation can be derived from the submitted input, keep the query
cache authoritative and project the latest pending mutation over it. This is the Effect Query
equivalent of TanStack Query v5's simplified optimistic-update pattern: the mutation input is
provisional intent, while the query result is confirmed state.

```tsx
const userScope = Mutation.MutationScope("user:1")
const pendingNamesAtom = Mutation.state({
  filters: {
    mutation: renameUser,
    scope: userScope,
    status: "pending"
  },
  select: ({ input }) => input.name
})

function UserName() {
  const user = useAtomValue(client.query(userQuery, { id: "1" }))
  const pendingNames = useAtomValue(pendingNamesAtom)

  if (!Result.isSuccess(user.result)) return <p>Loading…</p>

  const presentedName = pendingNames.at(-1) ?? user.result.value.name
  return <p>{presentedName}</p>
}
```

The presentation changes as soon as the mutation invocation enters the cache. If the command
fails, that invocation stops matching the pending filter and the presentation falls back to the
authoritative query value. If the command succeeds, `synchronize` makes the accepted value visible
through the query before the mutation stops being pending, so the presentation does not flicker
back to stale data.

This pattern requires no `QueryClient.setData`, snapshot, or rollback. Use a cache update instead
only when consumers must observe a speculative query result and the mutation can correctly derive
that complete result. Mutation state describes the submitted command; resource lifecycle and
confirmed state continue to come from queries.
