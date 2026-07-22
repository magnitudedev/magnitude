import * as HttpClient from "@effect/platform/HttpClient"
import { RpcClient, RpcClientError } from "@effect/rpc"
import { Effect, Layer } from "effect"
import {
  isInterruptedExit,
  makeJitDaemonCoordinator,
  recoveringProtocolLayer as jitRecoveringProtocolLayer,
} from "../jit-rpc"
import { DaemonSpawnerTag, toJitDaemonProvider } from "./daemon-spawner"
import type { DaemonError } from "./errors"
import { acnSubscriptionProtocol } from "./acn-subscription-protocol"

export interface AcnJitRuntimeOptions {
  /** Explicit ACN spawn command; when omitted the spawner resolves the binary. */
  readonly spawnCommand?: string[]
}

/** One process-local ACN runtime shared by every RPC consumer. */
export interface AcnJitRuntime {
  readonly protocolLayer: Layer.Layer<RpcClient.Protocol, never, HttpClient.HttpClient>
}

const { RpcClientError: TransportError } = RpcClientError

const unavailableError = (cause: DaemonError): RpcClientError.RpcClientError =>
  new TransportError({
    reason: "Unknown",
    message: `ACN unavailable: ${cause._tag}`,
    cause,
  })

/**
 * The sole ACN JIT composition entrypoint. It creates one coordinator,
 * performs startup demand once, and returns one protocol layer for every RPC
 * consumer. Subscription framing remains inside the ACN adapter.
 */
export const makeAcnJitRuntime = (
  options?: AcnJitRuntimeOptions,
): Effect.Effect<AcnJitRuntime, RpcClientError.RpcClientError, DaemonSpawnerTag> =>
  Effect.gen(function* () {
    const spawner = yield* DaemonSpawnerTag
    const coordinator = yield* makeJitDaemonCoordinator(
      toJitDaemonProvider(spawner, options?.spawnCommand),
    )
    yield* coordinator.ensure.pipe(Effect.mapError(unavailableError))

    return {
      protocolLayer: jitRecoveringProtocolLayer({
        coordinator,
        rpcPath: "/rpc",
        streamProtocol: acnSubscriptionProtocol,
        isEndpointRetirementExit: isInterruptedExit,
        classifyInfraError: unavailableError,
      }),
    }
  })
