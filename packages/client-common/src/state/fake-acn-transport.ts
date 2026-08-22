/**
 * Test transport for the ACN contract: routes each Rpc by tag to a handler.
 * Result types are owned by the Rpc, so the handler is untyped by design.
 */
import { Effect, Stream } from "effect"
import type { AcnTransport } from "@magnitudedev/sdk"

export type FakeRequestHandler = (tag: string, payload: unknown) => Effect.Effect<unknown, unknown>
export type FakeStreamHandler = (tag: string, payload: unknown) => Stream.Stream<unknown, unknown>

export const fakeAcnTransport = (
  request: FakeRequestHandler,
  stream: FakeStreamHandler = () => Stream.never,
): AcnTransport => ({
  boundary: "Acn",
  request: (rpc, payload) => request(rpc._tag, payload) as never,
  stream: (rpc, payload) => stream(rpc._tag, payload) as never,
})
