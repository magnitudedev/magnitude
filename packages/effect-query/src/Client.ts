import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import type * as Reactivity from "@effect/experimental/Reactivity"
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import * as Group from "./Group.js"
import * as Mutation from "./Mutation.js"
import * as Operation from "./Operation.js"
import * as Query from "./Query.js"
import * as QueryClient from "./QueryClient.js"
import * as Subscription from "./Subscription.js"

export interface Client<Provided, RuntimeError> {
  readonly runtime: Atom.AtomRuntime<Provided | QueryClient.QueryClient, RuntimeError>
  readonly query: <Input, Data, Error, Required extends Provided | QueryClient.QueryClient | Reactivity.Reactivity>(
    definition: Query.Query<Input, Data, Error, Required>,
    input: Input
  ) => Query.QueryAtom<Input, Data, Error | RuntimeError | Operation.ImplementationError<Provided>, Required>
  readonly mutation: <Input, Output, CommandError, Required extends Provided | QueryClient.QueryClient | Reactivity.Reactivity, SynchronizationError>(
    definition: Mutation.Mutation<Input, Output, CommandError, Required, SynchronizationError>
  ) => Mutation.MutationAtom<Input, Output, CommandError | RuntimeError | Operation.ImplementationError<Provided>, Required, SynchronizationError>
  readonly subscription: <Input, Event, Error, Required extends Provided | QueryClient.QueryClient | Reactivity.Reactivity>(
    definition: Subscription.Subscription<Input, Event, Error, Required>,
    input: Input
  ) => Subscription.SubscriptionAtom<Input, Event, Error | RuntimeError | Operation.ImplementationError<Provided>, Required>
}

/**
 * A group's members as seen through a client: a query member is the function
 * producing its canonical query atom, a mutation member is its canonical
 * mutation atom, a subscription member is the function producing its canonical
 * subscription atom, and a nested group is its materialized members. Each is
 * exactly what the corresponding materializer returns for that definition.
 */
type MaterializedMembers<Members, Provided, RuntimeError> = {
  readonly [Name in keyof Members]:
    Members[Name] extends Group.Any
      ? MaterializedMembers<Group.Members<Members[Name]>, Provided, RuntimeError>
    : Members[Name] extends Query.Query<infer Input, infer Data, infer Error, infer Requirements>
      ? (input: Input) => Query.QueryAtom<Input, Data, Error | RuntimeError | Operation.ImplementationError<Provided>, Requirements>
    : Members[Name] extends Mutation.Mutation<infer Input, infer Output, infer CommandError, infer Requirements, infer SynchronizationError>
      ? Mutation.MutationAtom<Input, Output, CommandError | RuntimeError | Operation.ImplementationError<Provided>, Requirements, SynchronizationError>
    : Members[Name] extends Subscription.Subscription<infer Input, infer Event, infer Error, infer Requirements>
      ? (input: Input) => Subscription.SubscriptionAtom<Input, Event, Error | RuntimeError | Operation.ImplementationError<Provided>, Requirements>
    : never
}

/** The members of group `G`, materialized by a client providing `Provided`. */
export type Materialized<G extends Group.Any, Provided, RuntimeError> =
  MaterializedMembers<Group.Members<G>, Provided, RuntimeError>

/** A client made for a group: the materializers plus every member of the group, materialized, at its name. */
export type GroupClient<G extends Group.Any, Provided, RuntimeError> =
  Client<Provided, RuntimeError> & Materialized<G, Provided, RuntimeError>

type Reserved = keyof Client<never, never>
type Requirements<Op> = Query.Requirements<Op> | Mutation.Requirements<Op> | Subscription.Requirements<Op>

interface NotMaterializable<Reason extends string> {
  readonly "@magnitudedev/effect-query/Client/NotMaterializable": Reason
}

/**
 * `unknown` when a client providing `Provided` can materialize every operation of
 * `G` — the constraint `query`/`mutation`/`subscription` impose per call, applied
 * once at construction — and no top-level member is named like a materializer.
 */
export type Materializable<G extends Group.Any, Provided> =
  [keyof Group.Members<G> & Reserved] extends [never]
    ? [Requirements<Group.Definition<G>>] extends [Provided | QueryClient.QueryClient | Reactivity.Reactivity]
      ? unknown
      : NotMaterializable<"an operation of the group requires a service the client does not provide">
    : NotMaterializable<"a member of the group is named runtime, query, mutation, or subscription">

type AdditionalLayer<Client_, Provided, Additional, AdditionalError> = (client: Client_) => Layer.Layer<
  Additional,
  AdditionalError,
  Provided | QueryClient.QueryClient | AtomRegistry.AtomRegistry | Reactivity.Reactivity
>

export function make<Provided, RuntimeError, Additional = never, AdditionalError = never>(
  services: Layer.Layer<Provided, RuntimeError, never>,
  additional?: AdditionalLayer<Client<Provided | Additional, RuntimeError | AdditionalError>, Provided, Additional, AdditionalError>,
): Client<Provided | Additional, RuntimeError | AdditionalError>
export function make<G extends Group.Any, Provided, RuntimeError, Additional = never, AdditionalError = never>(
  group: G & Materializable<G, Provided | Additional>,
  services: Layer.Layer<Provided, RuntimeError, never>,
  additional?: AdditionalLayer<GroupClient<G, Provided | Additional, RuntimeError | AdditionalError>, Provided, Additional, AdditionalError>,
): GroupClient<G, Provided | Additional, RuntimeError | AdditionalError>
export function make(...args: ReadonlyArray<unknown>): never {
  const [group, services, additional] = Group.isGroup(args[0])
    ? [args[0], args[1], args[2]]
    : [undefined, args[0], args[1]]
  if (!isServices(services)) throw new TypeError("Client.make: services must be a Layer")
  if (additional !== undefined && typeof additional !== "function") {
    throw new TypeError("Client.make: additional must be a function of the client")
  }
  return build(services, additional as ErasedAdditional | undefined, group) as never
}

type ErasedClient = Client<unknown, unknown>
type ErasedAdditional = (client: ErasedClient) => Layer.Layer<unknown, unknown, unknown>

/** The overloads admit only fully provided service Layers. */
const isServices = (value: unknown): value is Layer.Layer<unknown, unknown, never> => Layer.isLayer(value)

const build = (
  services: Layer.Layer<unknown, unknown, never>,
  additional: ErasedAdditional | undefined,
  group: Group.Any | undefined,
): ErasedClient => {
  let runtime!: ErasedClient["runtime"]
  const queryFamilies = new WeakMap<Query.Any, (input: unknown) => Query.QueryAtom<unknown, unknown, unknown, unknown>>()
  const mutations = new WeakMap<Mutation.Any, Mutation.MutationAtom<unknown, unknown, unknown, unknown, unknown>>()
  const subscriptionFamilies = new WeakMap<
    Subscription.Any,
    (input: unknown) => Subscription.SubscriptionAtom<unknown, unknown, unknown, unknown>
  >()

  const query: ErasedClient["query"] = (definition, input) => {
    let family = queryFamilies.get(definition)
    if (family === undefined) {
      family = Query.makeAtomFamily(runtime, definition as never)
      queryFamilies.set(definition, family)
    }
    return family(input) as never
  }

  const mutation: ErasedClient["mutation"] = (definition) => {
    let atom = mutations.get(definition)
    if (atom === undefined) {
      atom = Mutation.makeAtom(runtime, definition as never)
      mutations.set(definition, atom)
    }
    return atom as never
  }

  const subscription: ErasedClient["subscription"] = (definition, input) => {
    let family = subscriptionFamilies.get(definition)
    if (family === undefined) {
      family = Subscription.makeAtomFamily(runtime, definition as never)
      subscriptionFamilies.set(definition, family)
    }
    return family(input) as never
  }

  /**
   * Members materialize through the client's own memoized materializers, so a
   * member and the materializer return the same canonical atom. Mutation members
   * are accessors: every member materializes on first use, after the runtime
   * exists.
   */
  const materializeInto = (target: object, members: Group.Any): void => {
    for (const [name, member] of Object.entries(members)) {
      if (Object.hasOwn(target, name)) {
        throw new TypeError(`Client.make: group member ${name} collides with a client property`)
      }
      if (Group.isGroup(member)) {
        const nested = {}
        materializeInto(nested, member)
        Object.defineProperty(target, name, { enumerable: true, value: Object.freeze(nested) })
      } else if (Query.isQuery(member)) {
        Object.defineProperty(target, name, { enumerable: true, value: (input: unknown) => query(member as never, input) })
      } else if (Mutation.isMutation(member)) {
        Object.defineProperty(target, name, { enumerable: true, get: () => mutation(member as never) })
      } else if (Subscription.isSubscription(member)) {
        Object.defineProperty(target, name, { enumerable: true, value: (input: unknown) => subscription(member as never, input) })
      } else {
        throw new TypeError(`Client.make: group member ${name} is not a Query, Mutation, Subscription, or Group`)
      }
    }
  }

  const client: ErasedClient = {
    get runtime() {
      return runtime
    },
    query,
    mutation,
    subscription
  }
  if (group !== undefined) materializeInto(client, group)

  const infrastructure = Layer.mergeAll(
    services,
    QueryClient.makeLayer(query as QueryClient.Service["resolve"]),
  )
  const layer = (additional === undefined
    ? infrastructure
    : additional(client).pipe(Layer.provideMerge(infrastructure))) as Layer.Layer<
      unknown,
      unknown,
      AtomRegistry.AtomRegistry | Reactivity.Reactivity
    >
  const runtimeFactory = Atom.context({ memoMap: Effect.runSync(Layer.makeMemoMap) })
  runtime = Atom.keepAlive(runtimeFactory(layer))

  return client
}
