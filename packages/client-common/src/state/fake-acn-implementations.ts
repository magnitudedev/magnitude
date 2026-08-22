import type { RpcClientError } from "@effect/rpc/RpcClientError"
import { Effect, Layer, Stream } from "effect"
import { Operation } from "@magnitudedev/effect-query"

export type FakeRequestHandler = (name: string, payload: unknown) => Effect.Effect<unknown, unknown>
export type FakeStreamHandler = (name: string, payload: unknown) => Stream.Stream<unknown, unknown>

/** Test-only in-process implementations for declarative ACN operations. */
export const fakeAcnImplementationsLayer = (
  execute: FakeRequestHandler,
  stream: FakeStreamHandler = () => Stream.never,
) => Layer.succeed(
  Operation.implementationsTag<RpcClientError>(),
  {
    execute: (operation, payload) => execute(Operation.declaration(operation).name, payload),
    stream: (operation, payload) => stream(Operation.declaration(operation).name, payload),
  },
)
