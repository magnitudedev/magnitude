# `@magnitudedev/effect-query`

Effect-native query and mutation state built on Effect Atom. Definitions retain their exact input,
data, expected-error, and service-requirement types.

## Queries

Bind definitions to the Atom runtime that provides their Effect services, then define a canonical
key and fetch Effect:

```ts
import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import * as AtomResult from "@effect-atom/atom/Result"
import * as Data from "effect/Data"
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import * as Option from "effect/Option"
import { Query, QueryClient } from "@magnitudedev/effect-query"

class UserNotFound extends Data.TaggedError("UserNotFound")<{
  readonly id: string
}> {}

const AppQuery = Query.bind(Atom.runtime(Layer.empty))

const userQuery = AppQuery.make("User", {
  key: ({ id }: { readonly id: string }) => Data.tuple(id),
  effect: ({ id }) => id === "missing"
    ? Effect.fail(new UserNotFound({ id }))
    : Effect.succeed({ id, name: "Ada" }),
  staleTime: "30 seconds",
  gcTime: "5 minutes"
})

const registry = AtomRegistry.make()
const user = userQuery({ id: "1" })

const program = QueryClient.fetch(user).pipe(
  Effect.provide(QueryClient.layer),
  Effect.provideService(AtomRegistry.AtomRegistry, registry)
)

const value = await Effect.runPromise(program)
const state = registry.get(user)
const current = AtomResult.value(state.result)
```

Equivalent keys return the same query atom and share cached data and in-flight work. Structured keys
must be Effect `Data` or another value implementing Effect `Equal`; plain objects and arrays are
rejected because they have reference identity.

`QueryClient` also provides exact operations for cache control:

```ts
const cacheProgram = Effect.gen(function*() {
  yield* QueryClient.ensure(user)                  // retained value, revalidate if stale
  yield* QueryClient.prefetch(user)                // failure stays in query state
  yield* QueryClient.invalidate(userQuery.match())
  yield* QueryClient.invalidate(userQuery.match({ id: "1" }), { refetch: false })
  yield* QueryClient.cancel({ stale: true })
  yield* QueryClient.setData(user, (current) => Option.match(current, {
    onNone: () => ({ id: "1", name: "Grace" }),
    onSome: (value) => ({ ...value, name: "Grace" })
  }))
  yield* QueryClient.remove(userQuery.match({ id: "1" }))
})
```

`fetch` and `ensure` preserve the query's exact error type. Broad filtered `refetch`
instead returns `QueryBatchError`, because matches may have unrelated error types.

## RPC-backed definitions

An Effect Atom RPC tag already owns the typed RPC client Layer and Atom runtime. Use that runtime to
execute RPC Effects inside query definitions; do not call `Acn.query`, which would introduce a
second cache:

```ts
import * as AtomRpc from "@effect-atom/atom/AtomRpc"
import { FetchHttpClient } from "@effect/platform"
import { RpcClient, RpcSerialization } from "@effect/rpc"
import { MagnitudeRpcs } from "./protocol.js"

class AcnClient {}

const Acn = AtomRpc.Tag<AcnClient>()("AcnClient", {
  group: MagnitudeRpcs,
  protocol: RpcClient.layerProtocolHttp({ url: "http://127.0.0.1:3030/rpc" }).pipe(
    Layer.provide(RpcSerialization.layerNdjson),
    Layer.provide(FetchHttpClient.layer)
  )
})

const AcnQuery = Query.bind(Acn.runtime)

const sessionQuery = AcnQuery.make("Session", {
  key: ({ sessionId }: { readonly sessionId: string }) => Data.tuple(sessionId),
  effect: (payload) => Effect.flatMap(Acn, (client) =>
    client("GetSession", payload))
})
```

Commands use the same client service. Include `QueryClient.layer` in the mutation runtime when the
synchronization Effect updates cached queries:

```ts
const AcnMutation = Mutation.bind(Atom.runtime(
  Layer.merge(Acn.layer, QueryClient.layer)
))

const deleteSession = AcnMutation.make("DeleteSession", {
  effect: (payload: { readonly sessionId: string }) =>
    Effect.flatMap(Acn, (client) => client("DeleteSession", payload)),
  synchronize: (_output, payload) =>
    QueryClient.remove(sessionQuery.match(payload)),
  scope: ({ sessionId }) => Mutation.MutationScope(`session:${sessionId}`)
})
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

const IcnQuery = Query.bind(IcnHttp.runtime)

const hardwareQuery = IcnQuery.make("Hardware", {
  key: () => Data.tuple("hardware"),
  effect: () => Effect.flatMap(IcnHttp, (client) =>
    client.system.getHardware({})),
  staleTime: "10 seconds"
})

const modelDownloadQuery = IcnQuery.make("ModelDownload", {
  key: ({ attemptId }: { readonly attemptId: string }) => Data.tuple(attemptId),
  effect: ({ attemptId }) => Effect.flatMap(IcnHttp, (client) =>
    client.models.getModelDownload({ path: { attempt_id: attemptId } }))
})
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
  const client = yield* Acn

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
const renameModel = AcnMutation.make("RenameModel", {
  effect: (payload: {
    readonly modelId: string
    readonly name: string
  }) => Effect.flatMap(Acn, (client) =>
    client("RenameModel", payload)),

  synchronize: (_receipt, payload) =>
    QueryClient.fetch(modelQuery({ modelId: payload.modelId })).pipe(Effect.asVoid)
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
  const state = useAtomValue(sessionQuery({ sessionId }))
  const remove = useAtomSet(deleteSession, { mode: "promise" })

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

`RegistryProvider` owns the registry used by observation, remote-client runtimes, and
`QueryClient.layer`, so reads, commands, and synchronization operate on the same cache.

## Mutations

Mutation command and synchronization Effects retain separate error and requirement types. Bind a
runtime containing `QueryClient` when synchronization updates query state:

```ts
import { Mutation } from "@magnitudedev/effect-query"

const AppMutation = Mutation.bind(Atom.runtime(QueryClient.layer))

const renameUser = AppMutation.make("RenameUser", {
  effect: ({ id, name }: { readonly id: string; readonly name: string }) =>
    Effect.succeed({ id, name }),
  synchronize: (_output, input) =>
    QueryClient.invalidate(userQuery.match({ id: input.id })),
  scope: ({ id }) => Mutation.MutationScope(`user:${id}`)
})

const renameProgram = Mutation.execute(renameUser, { id: "1", name: "Grace" }).pipe(
  Effect.provideService(AtomRegistry.AtomRegistry, registry)
)
```

Executions with the same scope are serialized. A synchronization failure is reported as
`MutationSynchronizationError<Output, SynchronizationError>`, retaining both the accepted command
output and the exact visibility error. Reactive aggregate state is available through
`QueryClient.isFetching`, `QueryClient.isMutating`, and `QueryClient.mutationState`.
