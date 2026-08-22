/** Effect RPC projection for transport-neutral Effect Query operation groups. */
import * as Rpc from "@effect/rpc/Rpc"
import * as RpcClient from "@effect/rpc/RpcClient"
import type { RpcClientError } from "@effect/rpc/RpcClientError"
import * as RpcGroup from "@effect/rpc/RpcGroup"
import type * as RpcMiddleware from "@effect/rpc/RpcMiddleware"
import * as RpcServer from "@effect/rpc/RpcServer"
import * as Context from "effect/Context"
import * as Effect from "effect/Effect"
import * as Layer from "effect/Layer"
import type * as Schema from "effect/Schema"
import type * as Scope from "effect/Scope"
import * as CoreGroup from "../Group.js"
import * as Operation from "../Operation.js"

type MadeRpc<
  Name extends string,
  Payload extends Operation.PayloadInput,
  Success extends Schema.Schema.Any,
  Error extends Schema.Schema.All,
  Stream extends boolean,
> = ReturnType<typeof Rpc.make<Name, Payload, Success, Error, Stream>>

type WithMiddleware<R extends Rpc.Any, Middleware extends RpcMiddleware.TagClassAny> =
  [Middleware] extends [never] ? R : Rpc.AddMiddleware<R, Middleware>

type RpcOf<
  Value extends Operation.Any,
  FiniteMiddleware extends RpcMiddleware.TagClassAny,
> = Value extends Operation.Any
  ? Operation.OperationKind<Value> extends "subscription" | "queryFromStream"
    ? MadeRpc<Operation.Name<Value>, Operation.Payload<Value>, Operation.SuccessSchema<Value>, Operation.ErrorSchema<Value>, true>
    : WithMiddleware<
        MadeRpc<Operation.Name<Value>, Operation.Payload<Value>, Operation.SuccessSchema<Value>, Operation.ErrorSchema<Value>, false>,
        FiniteMiddleware
      >
  : never

export type GroupRpcs<
  Value extends CoreGroup.Any,
  FiniteMiddleware extends RpcMiddleware.TagClassAny = never,
> = RpcOf<CoreGroup.Operation<Value>, FiniteMiddleware>

/** Stable derived facts exposed to transport integrations without manual RPC declarations. */
export interface OperationMetadata {
  readonly name: string
  readonly kind: Operation.Kind
  readonly stream: boolean
  readonly annotations: Context.Context<never>
}

export interface MakeOptions<FiniteMiddleware extends RpcMiddleware.TagClassAny> {
  readonly decorate?: (operation: Operation.Any, rpc: Rpc.AnyWithProps) => unknown
}

interface Projection<Rpcs extends Rpc.Any> {
  readonly group: RpcGroup.RpcGroup<Rpcs>
  readonly byOperation: ReadonlyMap<Operation.Any, Rpc.Any>
  readonly metadata: ReadonlyMap<string, OperationMetadata>
}

export interface Adapter<FiniteMiddleware extends RpcMiddleware.TagClassAny = never> {
  /** Derive the Effect RPC group. Intended for low-level protocol and test integration. */
  readonly toRpcGroup: <Value extends CoreGroup.Any>(group: Value) => RpcGroup.RpcGroup<GroupRpcs<Value, FiniteMiddleware>>
  readonly implementations: <Value extends CoreGroup.Any>(
    group: Value,
    client: RpcClient.RpcClient.Flat<GroupRpcs<Value, FiniteMiddleware>, RpcClientError>,
  ) => Operation.ImplementationService<RpcClientError>
  readonly layer: <Value extends CoreGroup.Any>(group: Value) => Layer.Layer<
    Operation.Implementations<RpcClientError>,
    never,
    RpcClient.Protocol | Rpc.MiddlewareClient<GroupRpcs<Value, FiniteMiddleware>> | Rpc.Context<GroupRpcs<Value, FiniteMiddleware>>
  >
  readonly makeRpcClient: <Value extends CoreGroup.Any>(group: Value) => Effect.Effect<
    RpcClient.RpcClient<GroupRpcs<Value, FiniteMiddleware>, RpcClientError>,
    never,
    RpcClient.Protocol | Rpc.MiddlewareClient<GroupRpcs<Value, FiniteMiddleware>> | Scope.Scope
  >
  readonly toLayer: <
    Value extends CoreGroup.Any,
    Rpcs extends GroupRpcs<Value, FiniteMiddleware>,
    Handlers extends RpcGroup.HandlersFrom<Rpcs>,
    Error = never,
    Requirements = never,
  >(
    group: Value,
    handlers: Handlers | Effect.Effect<Handlers, Error, Requirements>,
  ) => Layer.Layer<Rpc.ToHandler<Rpcs>, Error, Exclude<Requirements, Scope.Scope> | RpcGroup.HandlersContext<Rpcs, Handlers>>
  readonly makeRpcServer: <Value extends CoreGroup.Any>(group: Value) => Effect.Effect<
    never,
    never,
    RpcServer.Protocol | Rpc.ToHandler<GroupRpcs<Value, FiniteMiddleware>> | Rpc.Middleware<GroupRpcs<Value, FiniteMiddleware>> | Rpc.Context<GroupRpcs<Value, FiniteMiddleware>>
  >
  readonly operations: <Value extends CoreGroup.Any>(group: Value) => ReadonlyArray<OperationMetadata>
  readonly operation: <Value extends CoreGroup.Any>(group: Value, name: string) => OperationMetadata | undefined
}

const annotate = <R extends Rpc.AnyWithProps>(rpc: R, annotations: Context.Context<never>): R =>
  (rpc as unknown as { annotateContext: (context: Context.Context<never>) => R }).annotateContext(annotations)

export const make = <FiniteMiddleware extends RpcMiddleware.TagClassAny = never>(
  options?: MakeOptions<FiniteMiddleware>,
): Adapter<FiniteMiddleware> => {
  const projections = new WeakMap<object, Projection<Rpc.Any>>()

  const project = <Value extends CoreGroup.Any>(group: Value): Projection<GroupRpcs<Value, FiniteMiddleware>> => {
    const cached = projections.get(group)
    if (cached !== undefined) return cached as unknown as Projection<GroupRpcs<Value, FiniteMiddleware>>

    const rpcs: Rpc.Any[] = []
    const byOperation = new Map<Operation.Any, Rpc.Any>()
    const metadata = new Map<string, OperationMetadata>()
    for (const operation of CoreGroup.operations(group)) {
      const declaration = Operation.declaration(operation)
      const stream = declaration.kind === "subscription" || declaration.kind === "queryFromStream"
      const base = annotate(Rpc.make(declaration.name, {
        payload: declaration.payload,
        success: declaration.success,
        error: declaration.error,
        stream,
      }), declaration.annotations)
      const rpc = (options?.decorate?.(operation, base) ?? base) as Rpc.Any & Rpc.AnyWithProps
      rpcs.push(rpc)
      byOperation.set(operation, rpc)
      metadata.set(declaration.name, {
        name: declaration.name,
        kind: declaration.kind,
        stream,
        annotations: rpc.annotations,
      })
    }
    const projection: Projection<Rpc.Any> = {
      group: RpcGroup.make(...rpcs),
      byOperation,
      metadata,
    }
    projections.set(group, projection)
    return projection as unknown as Projection<GroupRpcs<Value, FiniteMiddleware>>
  }

  const implementations = <Value extends CoreGroup.Any>(
    group: Value,
    client: RpcClient.RpcClient.Flat<GroupRpcs<Value, FiniteMiddleware>, RpcClientError>,
  ): Operation.ImplementationService<RpcClientError> => {
    const projection = project(group)
    return {
      execute: (operation, payload) => {
        const rpc = projection.byOperation.get(operation)
        if (rpc === undefined) return Effect.dieMessage("Operation is not part of this RPC group")
        return client(rpc._tag as never, payload as never) as never
      },
      stream: (operation, payload) => {
        const rpc = projection.byOperation.get(operation)
        if (rpc === undefined) return Effect.dieMessage("Operation is not part of this RPC group") as never
        return client(rpc._tag as never, payload as never) as never
      },
    }
  }

  return {
    toRpcGroup: (group) => project(group).group,
    implementations,
    layer: (group) => {
      return Layer.scoped(
        Operation.implementationsTag<RpcClientError>(),
        Effect.map(RpcClient.make(project(group).group, { flatten: true }), (client) => implementations(group, client as never)),
      ) as never
    },
    makeRpcClient: (group) => RpcClient.make(project(group).group),
    toLayer: (group, handlers) => project(group).group.toLayer(handlers as never) as never,
    makeRpcServer: (group) => RpcServer.make(project(group).group) as never,
    operations: (group) => [...project(group).metadata.values()],
    operation: (group, name) => project(group).metadata.get(name),
  }
}
