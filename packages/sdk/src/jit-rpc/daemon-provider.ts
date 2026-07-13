import { Effect, Option } from "effect"

export interface JitDaemonEndpoint {
  readonly url: string
}

export interface JitDaemonProvider<E> {
  readonly discover: () => Effect.Effect<Option.Option<JitDaemonEndpoint>, E, never>
  readonly spawn: () => Effect.Effect<JitDaemonEndpoint, E, never>
}
