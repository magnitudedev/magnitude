import { FetchHttpClient } from "@effect/platform"
import * as HttpClient from "@effect/platform/HttpClient"
import { RpcClient, RpcClientError } from "@effect/rpc"
import { Effect, Layer } from "effect"
import type { Scope } from "effect/Scope"
import { MagnitudeRpcs } from "@magnitudedev/protocol"
import {
  makeJitDaemonResolver,
  recoveringProtocolLayer as jitRecoveringProtocolLayer,
} from "../jit-rpc"
import type { AcnClient } from "../protocol"
import { DaemonSpawnerTag, toJitDaemonProvider } from "./daemon-spawner"
import type { DaemonError } from "./errors"
import { acnResidentStreamPolicy, isEncodedHeartbeat } from "./acn-stream-policy"

export { isEncodedHeartbeat }

export interface RecoveringClientOptions {
  /** Explicit ACN spawn command; when omitted the spawner resolves the binary. */
  readonly spawnCommand?: string[]
}

const { RpcClientError: TransportError } = RpcClientError

const unavailableError = (cause: DaemonError): RpcClientError.RpcClientError =>
  new TransportError({
    reason: "Unknown",
    message: `ACN unavailable: ${cause._tag}`,
    cause,
  })

/**
 * ACN binding for generic executable-backed JIT RPC recovery.
 *
 * This layer supplies only Magnitude-specific facts: the daemon provider,
 * `/rpc` path, ACN resident-stream policy, and ACN infrastructure error
 * mapping. The recovery mechanics live in `jit-rpc`.
 */
export const recoveringProtocolLayer = (
  options?: RecoveringClientOptions,
): Layer.Layer<RpcClient.Protocol, never, DaemonSpawnerTag | HttpClient.HttpClient> =>
  Layer.unwrapEffect(
    Effect.gen(function* () {
      const spawner = yield* DaemonSpawnerTag
      const provider = toJitDaemonProvider(spawner, options?.spawnCommand)
      const resolver = makeJitDaemonResolver(provider)
      return jitRecoveringProtocolLayer({
        resolver,
        rpcPath: "/rpc",
        streamPolicy: acnResidentStreamPolicy,
        classifyInfraError: unavailableError,
      })
    }),
  )

/**
 * An `AcnClient` with the operation contract built in: fire commands and trust
 * that an ACN will exist. Operations fail only with domain errors or fatal
 * infrastructure errors surfaced as `RpcClientError`.
 */
export const makeRecoveringAcnClient = (
  options?: RecoveringClientOptions,
): Effect.Effect<AcnClient, never, Scope | DaemonSpawnerTag> =>
  RpcClient.make(MagnitudeRpcs).pipe(
    Effect.provide(
      recoveringProtocolLayer(options).pipe(
        Layer.provide(FetchHttpClient.layer),
      ),
    ),
  )
