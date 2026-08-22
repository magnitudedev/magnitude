import * as Atom from "@effect-atom/atom/Atom"
import * as AtomRegistry from "@effect-atom/atom/Registry"
import type * as Reactivity from "@effect/experimental/Reactivity"
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
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

export const make = <Provided, RuntimeError, Additional = never, AdditionalError = never>(
  services: Layer.Layer<Provided, RuntimeError, never>,
  additional?: (client: Client<Provided | Additional, RuntimeError | AdditionalError>) =>
    Layer.Layer<
      Additional,
      AdditionalError,
      Provided | QueryClient.QueryClient | AtomRegistry.AtomRegistry | Reactivity.Reactivity
    >,
): Client<Provided | Additional, RuntimeError | AdditionalError> => {
  let runtime!: Client<Provided | Additional, RuntimeError | AdditionalError>["runtime"]
  const queryFamilies = new WeakMap<Query.Query<any, any, any, any>, (input: any) => Query.QueryAtom<any, any, any, any>>()
  const mutations = new WeakMap<Mutation.Mutation<any, any, any, any, any>, Mutation.MutationAtom<any, any, any, any, any>>()
  const subscriptionFamilies = new WeakMap<
    Subscription.Subscription<any, any, any, any>,
    (input: any) => Subscription.SubscriptionAtom<any, any, any, any>
  >()

  const query: Client<Provided | Additional, RuntimeError | AdditionalError>["query"] = (definition, input) => {
    let family = queryFamilies.get(definition)
    if (family === undefined) {
      family = Query.makeAtomFamily(runtime, definition as never)
      queryFamilies.set(definition, family)
    }
    return family(input)
  }

  const subscription: Client<Provided | Additional, RuntimeError | AdditionalError>["subscription"] = (definition, input) => {
    let family = subscriptionFamilies.get(definition)
    if (family === undefined) {
      family = Subscription.makeAtomFamily(runtime, definition as never)
      subscriptionFamilies.set(definition, family)
    }
    return family(input)
  }

  const client = {
    get runtime() {
      return runtime
    },
    query,
    mutation: (definition) => {
      let atom = mutations.get(definition)
      if (atom === undefined) {
        atom = Mutation.makeAtom(runtime, definition as never)
        mutations.set(definition, atom)
      }
      return atom
    },
    subscription
  } as Client<Provided | Additional, RuntimeError | AdditionalError>

  const infrastructure = Layer.mergeAll(
    services,
    QueryClient.makeLayer(query as QueryClient.Service["resolve"]),
  )
  const layer = (additional === undefined
    ? infrastructure
    : additional(client).pipe(Layer.provideMerge(infrastructure))) as Layer.Layer<
      Provided | QueryClient.QueryClient | Additional,
      RuntimeError | AdditionalError,
      AtomRegistry.AtomRegistry | Reactivity.Reactivity
    >
  const runtimeFactory = Atom.context({ memoMap: Effect.runSync(Layer.makeMemoMap) })
  runtime = Atom.keepAlive(runtimeFactory(layer))

  return client
}
