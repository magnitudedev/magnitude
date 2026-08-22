/**
 * Effect RPC adapter for Effect Query.
 *
 * A boundary defines queries, mutations, and subscriptions. Each definition is
 * a core Effect Query definition (usable with `Client.query` / `Client.mutation`
 * / `Client.subscription` unchanged) that also carries its `@effect/rpc` `Rpc`,
 * so the wire group is assembled from the same values the client consumes.
 *
 * The only transport knowledge lives in `Transport`: a per-boundary service
 * that executes one Rpc and returns values typed by that Rpc. Anything that
 * can satisfy `Transport` (an `RpcClient`, an in-process handler, a test fake)
 * is a valid transport.
 */
import * as Rpc from "@effect/rpc/Rpc"
import * as RpcClient from "@effect/rpc/RpcClient"
import type { RpcClientError } from "@effect/rpc/RpcClientError"
import type * as RpcGroup from "@effect/rpc/RpcGroup"
import * as Context from "effect/Context"
import type * as Duration from "effect/Duration"
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import type * as Option from "effect/Option"
import type * as Schedule from "effect/Schedule"
import type * as Schema from "effect/Schema"
import * as Stream from "effect/Stream"
import * as Key from "../Key.js"
import type { MutationScope, QueryKey } from "../Model.js"
import * as Mutation from "../Mutation.js"
import * as Query from "../Query.js"
import * as Subscription from "../Subscription.js"

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

/** Executes Rpcs of one boundary. Result types come from the Rpc, never from the transport. */
export interface Transport<TransportError> {
  readonly request: <R extends Rpc.Any>(
    rpc: R,
    payload: Rpc.PayloadConstructor<R>
  ) => Effect.Effect<Rpc.Success<R>, Rpc.Error<R> | TransportError>
  readonly stream: <R extends Rpc.Any>(
    rpc: R,
    payload: Rpc.PayloadConstructor<R>
  ) => Stream.Stream<Rpc.SuccessChunk<R>, Rpc.ErrorExit<R> | TransportError>
}

/** The transport service of one boundary; the `Id` keeps boundaries distinct in the Effect context. */
export interface Client<Id extends string, TransportError> extends Transport<TransportError> {
  readonly boundary: Id
}

// ---------------------------------------------------------------------------
// Definitions
// ---------------------------------------------------------------------------

type PayloadInput = Schema.Schema.AnyNoContext | Schema.Struct.Fields
type MadeRpc<
  Tag extends string,
  Payload extends PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  IsStream extends boolean
> = ReturnType<typeof Rpc.make<Tag, Payload, Success, Error, IsStream>>

export interface QueryRpc<R extends Rpc.Any, Id extends string, TransportError> extends
  Query.Query<Rpc.PayloadConstructor<R>, Rpc.Success<R>, Rpc.Error<R> | TransportError, Client<Id, TransportError>>
{
  readonly rpc: R
}

export interface StreamQueryRpc<R extends Rpc.Any, Data, Id extends string, TransportError> extends
  Query.Query<Rpc.PayloadConstructor<R>, Data, Rpc.ErrorExit<R> | TransportError, Client<Id, TransportError>>
{
  readonly rpc: R
}

export interface MutationRpc<
  R extends Rpc.Any,
  Id extends string,
  TransportError,
  SynchronizationError,
  SynchronizationRequirements
> extends
  Mutation.Mutation<
    Rpc.PayloadConstructor<R>,
    Rpc.Success<R>,
    Rpc.Error<R> | TransportError,
    Client<Id, TransportError> | SynchronizationRequirements,
    SynchronizationError
  >
{
  readonly rpc: R
}

export interface SubscriptionRpc<R extends Rpc.Any, Id extends string, TransportError> extends
  Subscription.Subscription<
    Rpc.PayloadConstructor<R>,
    Rpc.SuccessChunk<R>,
    Rpc.ErrorExit<R> | TransportError,
    Client<Id, TransportError>
  >
{
  readonly rpc: R
}

export type AnyDefinition = { readonly rpc: Rpc.Any }

interface RpcShape<Payload extends PayloadInput, Success extends Schema.Schema.Any, Error extends Schema.Schema.All> {
  readonly payload?: Payload
  readonly success?: Success
  readonly error?: Error
  /** Runtime annotations merged onto the Rpc (transport metadata, recovery policy, …). */
  readonly annotations?: Context.Context<never>
}

export interface QueryOptions<Payload extends PayloadInput, Success extends Schema.Schema.Any, Error extends Schema.Schema.All, Input, Err>
  extends RpcShape<Payload, Success, Error>
{
  /** Cache identity; defaults to the canonical structural form of the constructed payload. */
  readonly key?: (input: Input) => QueryKey
  readonly staleTime?: Duration.DurationInput
  readonly gcTime?: Duration.DurationInput
  readonly retry?: Schedule.Schedule<unknown, Err, never>
  readonly refresh?: Schedule.Schedule<unknown, void, never>
}

export interface MutationOptions<
  Payload extends PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  Input,
  Output,
  Err,
  SynchronizationError,
  SynchronizationRequirements
> extends RpcShape<Payload, Success, Error> {
  readonly scope?: (input: Input) => MutationScope
  readonly synchronize?: (
    output: Output,
    input: Input
  ) => Effect.Effect<void, SynchronizationError, SynchronizationRequirements>
  readonly retry?: Schedule.Schedule<unknown, Err, never>
  readonly gcTime?: Duration.DurationInput
}

export interface SubscriptionOptions<Payload extends PayloadInput, Success extends Schema.Schema.Any, Error extends Schema.Schema.All, Input, Err>
  extends RpcShape<Payload, Success, Error>
{
  readonly key?: (input: Input) => QueryKey
  readonly reconnect?: Schedule.Schedule<unknown, Err, never>
  readonly gcTime?: Duration.DurationInput
}

export interface StreamQueryOptions<
  Payload extends PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  Input,
  Event,
  Data,
  Err
> extends RpcShape<Payload, Success, Error> {
  readonly key?: (input: Input) => QueryKey
  readonly reduce: (previous: Option.Option<Data>, event: Event) => Data
  readonly reconnect?: Schedule.Schedule<unknown, Err, never>
  readonly gcTime?: Duration.DurationInput
}

// ---------------------------------------------------------------------------
// Boundary
// ---------------------------------------------------------------------------

/** One RPC boundary: the transport service its definitions call, and the constructors that define them. */
export interface Boundary<Id extends string, TransportError> {
  readonly id: Id
  readonly Client: Context.Tag<Client<Id, TransportError>, Client<Id, TransportError>>

  readonly query: <
    const Tag extends string,
    Payload extends PayloadInput = typeof Schema.Void,
    Success extends Schema.Schema.Any = typeof Schema.Void,
    Error extends Schema.Schema.All = typeof Schema.Never
  >(
    tag: Tag,
    options: QueryOptions<
      Payload,
      Success,
      Error,
      Rpc.PayloadConstructor<MadeRpc<Tag, Payload, Success, Error, false>>,
      Rpc.Error<MadeRpc<Tag, Payload, Success, Error, false>> | TransportError
    >
  ) => QueryRpc<MadeRpc<Tag, Payload, Success, Error, false>, Id, TransportError>

  readonly mutation: <
    const Tag extends string,
    Payload extends PayloadInput = typeof Schema.Void,
    Success extends Schema.Schema.Any = typeof Schema.Void,
    Error extends Schema.Schema.All = typeof Schema.Never,
    SynchronizationError = never,
    SynchronizationRequirements = never
  >(
    tag: Tag,
    options: MutationOptions<
      Payload,
      Success,
      Error,
      Rpc.PayloadConstructor<MadeRpc<Tag, Payload, Success, Error, false>>,
      Rpc.Success<MadeRpc<Tag, Payload, Success, Error, false>>,
      Rpc.Error<MadeRpc<Tag, Payload, Success, Error, false>> | TransportError,
      SynchronizationError,
      SynchronizationRequirements
    >
  ) => MutationRpc<
    MadeRpc<Tag, Payload, Success, Error, false>,
    Id,
    TransportError,
    SynchronizationError,
    SynchronizationRequirements
  >

  readonly subscription: <
    const Tag extends string,
    Payload extends PayloadInput = typeof Schema.Void,
    Success extends Schema.Schema.Any = typeof Schema.Void,
    Error extends Schema.Schema.All = typeof Schema.Never
  >(
    tag: Tag,
    options: SubscriptionOptions<
      Payload,
      Success,
      Error,
      Rpc.PayloadConstructor<MadeRpc<Tag, Payload, Success, Error, true>>,
      Rpc.ErrorExit<MadeRpc<Tag, Payload, Success, Error, true>> | TransportError
    >
  ) => SubscriptionRpc<MadeRpc<Tag, Payload, Success, Error, true>, Id, TransportError>

  /** A query whose data is folded from a stream Rpc (first element = data, later elements reduced). */
  readonly queryFromStream: <
    const Tag extends string,
    Data,
    Payload extends PayloadInput = typeof Schema.Void,
    Success extends Schema.Schema.Any = typeof Schema.Void,
    Error extends Schema.Schema.All = typeof Schema.Never
  >(
    tag: Tag,
    options: StreamQueryOptions<
      Payload,
      Success,
      Error,
      Rpc.PayloadConstructor<MadeRpc<Tag, Payload, Success, Error, true>>,
      Rpc.SuccessChunk<MadeRpc<Tag, Payload, Success, Error, true>>,
      Data,
      Rpc.ErrorExit<MadeRpc<Tag, Payload, Success, Error, true>> | TransportError
    >
  ) => StreamQueryRpc<MadeRpc<Tag, Payload, Success, Error, true>, Data, Id, TransportError>

  /** Adapts a flat `RpcClient` of a group containing this boundary's Rpcs. */
  readonly transport: <Rpcs extends Rpc.Any>(
    client: RpcClient.RpcClient.Flat<Rpcs, TransportError>
  ) => Client<Id, TransportError>

  /** Builds the flat `RpcClient` for `group` and provides it as this boundary's `Client`. */
  readonly layer: <Rpcs extends Rpc.Any>(
    group: RpcGroup.RpcGroup<Rpcs>
  ) => Layer.Layer<
    Client<Id, TransportError>,
    never,
    RpcClient.Protocol | Rpc.MiddlewareClient<Rpcs> | Rpc.Context<Rpcs>
  >
}

interface Constructible {
  readonly make: (input: unknown) => unknown
}

const isConstructible = (schema: object): schema is Constructible =>
  "make" in schema && typeof schema.make === "function"

/** Canonical identity of a payload: the constructed (schema-normalized) value, structurally encoded. */
const payloadKey = (rpc: Rpc.AnyWithProps) => (input: unknown): QueryKey => {
  const schema: object = rpc.payloadSchema
  return Key.canonical(isConstructible(schema) ? schema.make(input) : input)
}

const annotated = <
  Tag extends string,
  Payload extends Schema.Schema.Any,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All
>(
  rpc: Rpc.Rpc<Tag, Payload, Success, Error>,
  annotations: Context.Context<never> | undefined
): Rpc.Rpc<Tag, Payload, Success, Error> =>
  annotations === undefined ? rpc : rpc.annotateContext(annotations)

export const make = <const Id extends string, TransportError = RpcClientError>(id: Id): Boundary<Id, TransportError> => {
  const Client = Context.GenericTag<Client<Id, TransportError>>(`@magnitudedev/effect-query/rpc/${id}`)

  const request = <R extends Rpc.Any>(rpc: R, input: Rpc.PayloadConstructor<R>) =>
    Effect.flatMap(Client, (client) => client.request(rpc, input))
  const stream = <R extends Rpc.Any>(rpc: R, input: Rpc.PayloadConstructor<R>) =>
    Stream.unwrap(Effect.map(Client, (client) => client.stream(rpc, input)))

  const transport: Boundary<Id, TransportError>["transport"] = (client) => ({
    boundary: id,
    // The flat client is defined for every Rpc of its group; the Rpc passed here
    // carries the types. This is the single typed/untyped seam of the adapter.
    request: (rpc, payload) => client(rpc._tag as never, payload as never) as never,
    stream: (rpc, payload) => client(rpc._tag as never, payload as never) as never
  })

  return {
    id,
    Client,
    query: (tag, options) => {
      const rpc = annotated(
        Rpc.make(tag, { payload: options.payload, success: options.success, error: options.error }),
        options.annotations
      )
      const definition = Query.make(tag, {
        key: options.key ?? payloadKey(rpc),
        staleTime: options.staleTime,
        gcTime: options.gcTime,
        retry: options.retry,
        refresh: options.refresh,
        effect: (input) => request(rpc, input)
      })
      return Object.assign(definition, { rpc })
    },
    mutation: (tag, options) => {
      const rpc = annotated(
        Rpc.make(tag, { payload: options.payload, success: options.success, error: options.error }),
        options.annotations
      )
      const definition = Mutation.make(tag, {
        effect: (input) => request(rpc, input),
        synchronize: options.synchronize,
        scope: options.scope,
        retry: options.retry,
        gcTime: options.gcTime
      })
      return Object.assign(definition, { rpc })
    },
    subscription: (tag, options) => {
      const rpc = annotated(
        Rpc.make(tag, {
          payload: options.payload,
          success: options.success,
          error: options.error,
          stream: true
        }),
        options.annotations
      )
      const definition = Subscription.make(tag, {
        key: options.key ?? payloadKey(rpc),
        stream: (input) => stream(rpc, input),
        reconnect: options.reconnect,
        gcTime: options.gcTime
      })
      return Object.assign(definition, { rpc })
    },
    queryFromStream: (tag, options) => {
      const rpc = annotated(
        Rpc.make(tag, {
          payload: options.payload,
          success: options.success,
          error: options.error,
          stream: true
        }),
        options.annotations
      )
      const definition = Query.fromStream(tag, {
        key: options.key ?? payloadKey(rpc),
        stream: (input) => stream(rpc, input),
        reduce: options.reduce,
        reconnect: options.reconnect,
        gcTime: options.gcTime
      })
      return Object.assign(definition, { rpc })
    },
    transport,
    layer: (group) => Layer.scoped(
      Client,
      Effect.map(RpcClient.make(group, { flatten: true }), (client) => transport(client))
    )
  }
}
