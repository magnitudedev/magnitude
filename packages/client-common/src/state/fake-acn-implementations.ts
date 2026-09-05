import { RpcSchema } from "@effect/rpc"
import { Effect, Layer, Stream } from "effect"
import { MagnitudeClient, MagnitudeRpcs, AcnRpcGroup, namespaceClient } from "@magnitudedev/sdk"

export type FakeRequestHandler = (name: string, payload: unknown) => Effect.Effect<unknown, unknown>
export type FakeStreamHandler = (name: string, payload: unknown) => Stream.Stream<unknown, unknown>

/** Test-only in-process implementations for declarative ACN operations. */
export const fakeAcnImplementationsLayer = (
  execute: FakeRequestHandler,
  stream: FakeStreamHandler = () => Stream.never,
) => {
  const operations = namespaceClient(MagnitudeRpcs, ((tag: string, payload: unknown) => {
    const rpc = AcnRpcGroup.requests.get(tag)
    if (!rpc) throw new TypeError(`Unknown test RPC: ${tag}`)
    return RpcSchema.isStreamSchema(rpc.successSchema) ? Stream.suspend(() => stream(tag, payload)) : Effect.suspend(() => execute(tag, payload))
  }) as Parameters<typeof namespaceClient<typeof MagnitudeRpcs, never>>[1])
  return Layer.succeed(MagnitudeClient, {
    ...operations,
    connection: {
      ...operations.connection,
      state: Effect.succeed({ _tag: "Idle" as const }),
      changes: Stream.empty,
      connect: Effect.void,
    },
  })
}
