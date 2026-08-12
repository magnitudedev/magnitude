import * as Atom from "@effect-atom/atom/Atom"
import type * as Reactivity from "@effect/experimental/Reactivity"
import * as Layer from "effect/Layer"
import * as Mutation from "./Mutation.js"
import * as Query from "./Query.js"
import * as QueryClient from "./QueryClient.js"

export interface Client<Provided, RuntimeError> {
  readonly runtime: Atom.AtomRuntime<Provided | QueryClient.QueryClient, RuntimeError>
  readonly query: <Input, Data, Error, Required extends Provided | QueryClient.QueryClient | Reactivity.Reactivity>(
    definition: Query.Query<Input, Data, Error, Required>,
    input: Input
  ) => Query.QueryAtom<Input, Data, Error | RuntimeError, Required>
  readonly mutation: <Input, Output, CommandError, Required extends Provided | QueryClient.QueryClient | Reactivity.Reactivity, SynchronizationError>(
    definition: Mutation.Mutation<Input, Output, CommandError, Required, SynchronizationError>
  ) => Mutation.MutationAtom<Input, Output, CommandError | RuntimeError, Required, SynchronizationError>
}

export const make = <Provided, RuntimeError>(
  services: Layer.Layer<Provided, RuntimeError, never>
): Client<Provided, RuntimeError> => {
  let runtime!: Client<Provided, RuntimeError>["runtime"]
  const queryFamilies = new WeakMap<Query.Query<any, any, any, any>, (input: any) => Query.QueryAtom<any, any, any, any>>()
  const mutations = new WeakMap<Mutation.Mutation<any, any, any, any, any>, Mutation.MutationAtom<any, any, any, any, any>>()

  const query: Client<Provided, RuntimeError>["query"] = (definition, input) => {
    let family = queryFamilies.get(definition)
    if (family === undefined) {
      family = Query.makeAtomFamily(runtime, definition as never)
      queryFamilies.set(definition, family)
    }
    return family(input)
  }

  runtime = Atom.runtime(Layer.mergeAll(
    services,
    QueryClient.makeLayer(query as QueryClient.Service["resolve"])
  ))

  return {
    runtime,
    query,
    mutation: (definition) => {
      let atom = mutations.get(definition)
      if (atom === undefined) {
        atom = Mutation.makeAtom(runtime, definition as never)
        mutations.set(definition, atom)
      }
      return atom
    }
  } as Client<Provided, RuntimeError>
}
