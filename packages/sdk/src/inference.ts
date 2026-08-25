import type { RpcClient } from "@effect/rpc"
import { RpcClientError } from "@effect/rpc/RpcClientError"
import { Effect, Layer } from "effect"
import { Operation } from "@magnitudedev/effect-query"
import { AcnBoundary, AcnRpc } from "@magnitudedev/acn-protocol"

/** The first-party client boundary. Every query, mutation, and subscription is served by ACN RPC. */
export const MagnitudeBoundary = AcnBoundary

export type MagnitudeImplementationError = RpcClientError

export const magnitudeImplementationsLayer = (
  protocolLayer: Layer.Layer<RpcClient.Protocol>,
): Layer.Layer<Operation.Implementations<MagnitudeImplementationError>> =>
  Layer.scoped(
    Operation.implementationsTag<MagnitudeImplementationError>(),
    AcnRpc.makeImplementations(AcnBoundary).pipe(Effect.provide(protocolLayer)),
  )
