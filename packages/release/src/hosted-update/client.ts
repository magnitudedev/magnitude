import { FetchHttpClient, HttpClient, HttpClientRequest } from "@effect/platform"
import { Clock, Effect, Option, Schema, Stream } from "effect"
import type { KeyObject } from "node:crypto"
import { newUpdateNonce, updateQuery, type UpdateSigningFailed } from "./request-auth"
import { UpdateRequest } from "./request"
import { acceptsUpdateManifest, verifyUpdateManifest, SignedUpdateManifest } from "./manifest"
import { releaseChannelOf } from "../client-update/release-channels"

export const UpdateClientMetadata = UpdateRequest.pipe(Schema.omit("protocol", "product", "channel", "ts", "nonce"))
export type UpdateClientMetadata = typeof UpdateClientMetadata.Type
export class HostedUpdateCheckFailed extends Schema.TaggedError<HostedUpdateCheckFailed>()("HostedUpdateCheckFailed", {
  reason: Schema.Literal("request", "network", "response", "publisher"),
}) {}

/** No retries or redirects. The scoped caller owns startup, hourly and explicit check admission. */
export const checkHostedUpdate = (options: {
  readonly origin: string
  readonly metadata: UpdateClientMetadata
  readonly sign: (url: URL) => Effect.Effect<string, UpdateSigningFailed>
  readonly trustedPublishers: ReadonlyMap<string, KeyObject>
  readonly userAgent: string
}) => Effect.gen(function* () {
  const channel = releaseChannelOf(options.metadata.version)
  const fields = yield* Schema.decodeUnknown(Schema.typeSchema(UpdateRequest))({
    ...options.metadata, protocol: "1", product: "desktop", channel: channel === "unknown" ? "stable" : channel,
    ts: Math.floor((yield* Clock.currentTimeMillis) / 1000), nonce: yield* newUpdateNonce,
  }).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "request" })))
  const encoded = yield* Schema.encode(UpdateRequest)(fields).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "request" })))
  const url = yield* Effect.try({ try: () => {
    const origin = new URL(options.origin)
    if (origin.origin !== options.origin || origin.protocol !== "https:") throw new Error("Invalid update origin")
    return new URL(`/api/update?${updateQuery(Object.fromEntries(Object.entries(encoded).filter((entry): entry is [string, string] => typeof entry[1] === "string")))}`, origin)
  }, catch: () => new HostedUpdateCheckFailed({ reason: "request" }) })
  const authorization = yield* options.sign(url).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "request" })))
  const http = (yield* HttpClient.HttpClient).pipe(HttpClient.withTracerDisabledWhen(() => true))
  const response = yield* http.execute(HttpClientRequest.get(url.href, {
    headers: { authorization, "user-agent": options.userAgent, accept: "application/json" },
  })).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "network" })))
  if (response.status === 204) return Option.none()
  if (response.status !== 200) return yield* new HostedUpdateCheckFailed({ reason: "response" })
  const bytes = yield* response.stream.pipe(Stream.runFoldEffect({ chunks: [] as Uint8Array[], size: 0 }, (state, chunk) => {
    const size = state.size + chunk.byteLength
    return size > 32 * 1024 ? new HostedUpdateCheckFailed({ reason: "response" })
      : Effect.succeed({ chunks: [...state.chunks, chunk], size })
  }), Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "response" })))
  const json = yield* Effect.try({ try: () => new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(bytes.chunks)), catch: () => new HostedUpdateCheckFailed({ reason: "response" }) })
  const envelope = yield* Schema.decodeUnknown(Schema.parseJson(SignedUpdateManifest))(json, { onExcessProperty: "error" }).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "response" })))
  const manifest = yield* verifyUpdateManifest(envelope, options.trustedPublishers).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "publisher" })))
  if (!acceptsUpdateManifest(manifest, fields)) return yield* new HostedUpdateCheckFailed({ reason: "response" })
  return Option.some(manifest)
}).pipe(
  Effect.provideService(FetchHttpClient.RequestInit, { redirect: "manual" }),
  Effect.timeoutFail({ duration: "10 seconds", onTimeout: () => new HostedUpdateCheckFailed({ reason: "network" }) }),
)
