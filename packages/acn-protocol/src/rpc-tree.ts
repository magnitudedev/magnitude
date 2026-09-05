import { Rpc, RpcGroup, RpcSchema } from "@effect/rpc";
import type { RpcClient } from "@effect/rpc";
import { Context, Effect, Option, Stream, type Schema } from "effect";
import {
  AcnRpcRecoveryPolicyTag,
  type RecoveryDeclared,
} from "./transport/recovery";

type DeclaredRpc = Rpc.Any &
  (
    | RecoveryDeclared
    | {
        readonly successSchema: Pick<RpcSchema.Stream<
          Schema.Schema.Any,
          Schema.Schema.All
        >, "success" | "failure" | "Type">;
      }
  );

/** A namespace contains ordinary RPC definitions, not another operation DSL. */
export interface RpcTree {
  readonly [key: string]: DeclaredRpc | RpcTree;
}

export type TreeRpcs<T> = T extends Rpc.Any
  ? T
  : T extends RpcTree
  ? { [K in keyof T]: TreeRpcs<T[K]> }[keyof T]
  : never;

export type RpcMethod<R extends Rpc.Any, Error> = (
  input: Rpc.PayloadConstructor<R>
) => Rpc.Success<R> extends Stream.Stream<infer A, infer E, infer Env>
  ? Stream.Stream<A, E | Rpc.Error<R> | Error, Env | Rpc.Context<R>>
  : Effect.Effect<Rpc.Success<R>, Rpc.Error<R> | Error, Rpc.Context<R>>;

export type TreeClient<T, Error> = {
  readonly [K in keyof T]: T[K] extends Rpc.Any
    ? RpcMethod<T[K], Error>
    : TreeClient<T[K], Error>;
};

export const rpcGroup = <const T extends RpcTree>(
  tree: T
): RpcGroup.RpcGroup<TreeRpcs<T>> => {
  const requests: Array<TreeRpcs<T>> = [];
  const tags = new Set<string>();
  const active = new Set<object>();
  const visit = (node: RpcTree): void => {
    if (active.has(node)) throw new TypeError("Cyclic RPC namespace");
    active.add(node);
    for (const [key, value] of Object.entries(node)) {
      if (Rpc.isRpc(value)) {
        if (tags.has(value._tag))
          throw new TypeError(`Duplicate RPC tag: ${value._tag}`);
        tags.add(value._tag);
        if (
          !RpcSchema.isStreamSchema(value.successSchema) &&
          Option.isNone(
            Context.getOption(value.annotations, AcnRpcRecoveryPolicyTag)
          )
        )
          throw new TypeError(`Missing RPC recovery policy: ${value._tag}`);
        // Every visited leaf belongs to T; dynamic traversal erases that key relationship.
        requests.push(value as TreeRpcs<T>);
      } else if (
        value !== null &&
        typeof value === "object" &&
        Object.getPrototypeOf(value) === Object.prototype
      ) {
        visit(value as RpcTree);
      } else throw new TypeError(`Invalid RPC namespace member: ${key}`);
    }
    active.delete(node);
  };
  visit(tree);
  return RpcGroup.make(...requests);
};

/** Map the authoritative namespace onto one Effect RPC client's dispatch function. */
export function namespaceClient<const T extends RpcTree, Error>(
  tree: T,
  client: RpcClient.RpcClient.Flat<TreeRpcs<T>, Error>
): TreeClient<T, Error>;
export function namespaceClient<const T extends RpcTree, Error, MappedError>(
  tree: T,
  client: RpcClient.RpcClient.Flat<TreeRpcs<T>, Error>,
  mapError: <E>(error: E | Error) => E | MappedError
): TreeClient<T, MappedError>;
export function namespaceClient<
  const T extends RpcTree,
  Error,
  MappedError = Error
>(
  tree: T,
  client: RpcClient.RpcClient.Flat<TreeRpcs<T>, Error>,
  mapError?: <E>(error: E | Error) => E | MappedError
): TreeClient<T, MappedError> {
  const visit = (node: RpcTree): object =>
    Object.fromEntries(
      Object.entries(node).map(([key, value]) => [
        key,
        Rpc.isRpc(value)
          ? (payload: unknown) => {
              const result = client(value._tag as never, payload as never);
              if (mapError === undefined) return result;
              return Effect.isEffect(result)
                ? Effect.mapError(result, mapError)
                : Stream.mapError(result, mapError);
            }
          : visit(value as RpcTree),
      ])
    );
  // The traversal preserves every key and associates it with that exact RPC tag.
  return visit(tree) as TreeClient<T, MappedError>;
}
