/**
 * Base protocol layer for known daemon URLs.
 *
 * This module is browser-safe: it builds the HTTP RPC protocol only and does
 * not spawn or discover services. Interactive clients use `makeAcnConnection`
 * from the SDK entrypoint; this layer serves tests and fixed-URL tooling.
 */
import { RpcClient, RpcSerialization } from "@effect/rpc"
import { FetchHttpClient } from "@effect/platform"
import { Layer } from "effect"

export { AcnBoundary, AcnRpc } from "@magnitudedev/acn-protocol"
export type * from "@magnitudedev/acn-protocol"

/**
 * Protocol layer for a known daemon URL. NDJSON over HTTP, no spawn.
 */
export const protocolLayer = (url: string) =>
  RpcClient.layerProtocolHttp({ url: `${url}/rpc` }).pipe(
    Layer.provide(RpcSerialization.layerNdjson),
    Layer.provide(FetchHttpClient.layer),
  )
