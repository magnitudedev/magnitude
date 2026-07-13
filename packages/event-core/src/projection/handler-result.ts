import { Effect } from 'effect'

export type ProjectionHandlerResult<State> =
  | State
  | Effect.Effect<State, unknown>

export const resolveProjectionHandlerResult = <State>(
  result: ProjectionHandlerResult<State>
): Effect.Effect<State, unknown> =>
  Effect.isEffect(result)
    ? result
    : Effect.succeed(result)
