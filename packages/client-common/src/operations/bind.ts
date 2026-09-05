import {
  Mutation,
  Operation,
  Query,
  Subscription,
} from "@magnitudedev/effect-query";
import { MagnitudeClient } from "@magnitudedev/sdk";
import { Effect, Stream } from "effect";

interface RpcIdentity {
  readonly _tag: string;
  readonly payloadSchema: Operation.PayloadInput;
}

/** Local cache bindings only: the SDK owns invocation, these callers own all cache policy. */
export const query = <I, A, E, R>(
  rpc: RpcIdentity,
  select: (client: MagnitudeClient) => (input: I) => Effect.Effect<A, E, R>,
  options: Omit<Query.Options<I, A, E, R>, "effect" | "key"> = {}
) =>
  Query.make(rpc._tag, {
    ...options,
    effect: (input: I) =>
      Effect.flatMap(MagnitudeClient, (client) => select(client)(input)),
    key: Operation.payloadKey(rpc.payloadSchema),
  });

export const mutation = <I, A, E, R, SE = never, SR = never>(
  rpc: RpcIdentity,
  select: (client: MagnitudeClient) => (input: I) => Effect.Effect<A, E, R>,
  options: Omit<Mutation.Options<I, A, E, R, SE, SR>, "effect"> = {}
) =>
  Mutation.make(rpc._tag, {
    ...options,
    effect: (input: I) =>
      Effect.flatMap(MagnitudeClient, (client) => select(client)(input)),
  });

export const subscription = <I, A, E, R>(
  rpc: RpcIdentity,
  select: (client: MagnitudeClient) => (input: I) => Stream.Stream<A, E, R>,
  options: Omit<Subscription.Options<I, A, E, R>, "stream" | "key"> = {}
) =>
  Subscription.make(rpc._tag, {
    ...options,
    stream: (input: I) =>
      Stream.unwrap(
        Effect.map(MagnitudeClient, (client) => select(client)(input))
      ),
    key: Operation.payloadKey(rpc.payloadSchema),
  });

export const streamQuery = <I, A, Data, E, R>(
  rpc: RpcIdentity,
  select: (client: MagnitudeClient) => (input: I) => Stream.Stream<A, E, R>,
  options: Omit<Query.FromStreamOptions<I, A, Data, E, R>, "stream" | "key">
) => Query.fromStream(rpc._tag, {
  ...options,
  stream: (input: I) => Stream.unwrap(Effect.map(MagnitudeClient, client => select(client)(input))),
  key: Operation.payloadKey(rpc.payloadSchema),
});
