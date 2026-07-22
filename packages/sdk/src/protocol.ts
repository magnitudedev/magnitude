/**
 * Base protocol layer for known daemon URLs.
 *
 * This module is browser-safe: it builds HTTP RPC transports only and does
 * not spawn or discover daemons. Consumers that need the operation contract
 * use `makeAcnJitRuntime` from the SDK entrypoint.
 */
import { RpcClient, RpcClientError, RpcSerialization } from "@effect/rpc"
import { FetchHttpClient } from "@effect/platform"
import { HttpClient } from "@effect/platform/HttpClient"
import { Context, Effect, Layer } from "effect"
import type { Scope } from "effect/Scope"
import { MagnitudeRpcs } from "@magnitudedev/protocol"

const { Protocol } = RpcClient

export { MagnitudeRpcs } from "@magnitudedev/protocol"
export type * from "@magnitudedev/protocol"

export type AcnClient = RpcClient.FromGroup<typeof MagnitudeRpcs, RpcClientError.RpcClientError>

export class AcnClientTag extends Context.Tag("AcnClient")<
  AcnClientTag,
  AcnClient
>() {
  /**
   * Connect-only layer for renderer/browser environments where the host
   * provides a daemon URL.
   */
  static readonly connectLayer = (url: string): Layer.Layer<AcnClientTag, never, HttpClient> =>
    Layer.scoped(this, connect(url))
}

/**
 * Protocol layer for a known daemon URL. NDJSON over HTTP, no spawn.
 */
export const protocolLayer = (url: string) =>
  RpcClient.layerProtocolHttp({ url: `${url}/rpc` }).pipe(
    Layer.provide(RpcSerialization.layerNdjson),
    Layer.provide(FetchHttpClient.layer),
  )

/**
 * Lightweight transport-error hook for fixed-URL clients. This does not
 * discover or spawn a daemon; it only lets UI code surface fixed URL failures.
 */
export const wrapProtocol = (
  onTransportError: () => void,
) =>
  Layer.scopedContext(
    Effect.gen(function* () {
      const protocol = yield* Protocol
      return Context.make(Protocol, {
        ...protocol,
        send: (request, transferables) =>
          protocol.send(request, transferables).pipe(
            Effect.tapError(() => Effect.sync(onTransportError)),
          ),
      })
    }),
  )

export const protocolLayerWithRecovery = (url: string, onTransportError: () => void) =>
  wrapProtocol(onTransportError).pipe(Layer.provide(protocolLayer(url)))

/**
 * Connect to a known daemon URL.
 */
export const connect = (
  url: string,
  onTransportError?: () => void,
): Effect.Effect<AcnClient, never, Scope> =>
  RpcClient.make(MagnitudeRpcs).pipe(
    Effect.provide(
      onTransportError
        ? protocolLayerWithRecovery(url, onTransportError)
        : protocolLayer(url),
    ),
  )

/**
 * Client layer for a known daemon URL.
 */
export const makeClientLayer = (url: string): Layer.Layer<AcnClientTag, never, never> =>
  Layer.scoped(AcnClientTag, RpcClient.make(MagnitudeRpcs)).pipe(
    Layer.provide(protocolLayer(url)),
  )
