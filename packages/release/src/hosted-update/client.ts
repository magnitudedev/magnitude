import { isGithubReleaseAssetUrl } from "./github-artifact"
import { FetchHttpClient, HttpClient, HttpClientRequest } from "@effect/platform"
import { Clock, Effect, Option, Schema, Stream } from "effect"
import type { KeyObject } from "node:crypto"
import { newUpdateNonce, updateQuery, type UpdateSigningFailed } from "./request-auth"
import { UpdateRequest, UpdateRequestFields, type UpdateCheckReason, type UpdateOutcome } from "./request"
import { acceptsUpdateRelease, verifyUpdateRelease, UpdateRelease, ReleaseTarget } from "./release"
import { releaseChannelOf } from "../client-update/release-channels"

export const UpdateClientMetadata = UpdateRequestFields.pipe(Schema.omit("protocol", "product", "channel", "ts", "nonce", "reason", "outcome", "outcome_version", "outcome_reason"))
export type UpdateClientMetadata = typeof UpdateClientMetadata.Type
export interface UpdateCheck {
  readonly reason: UpdateCheckReason
  readonly outcome: Option.Option<UpdateOutcome>
}
export class HostedUpdateCheckFailed extends Schema.TaggedError<HostedUpdateCheckFailed>()("HostedUpdateCheckFailed", {
  reason: Schema.Literal("request", "network", "response", "publisher"),
}) {}

export interface HostedUpdateConnection {
  readonly origin: string
  readonly metadata: UpdateClientMetadata
  readonly sign: (url: URL) => Effect.Effect<string, UpdateSigningFailed>
  readonly userAgent: string
}

const signedRequest = (options: HostedUpdateConnection, path: string, check: Option.Option<UpdateCheck>, extra: Readonly<Record<string, string>> = {}) => Effect.gen(function* () {
  const channel = releaseChannelOf(options.metadata.version)
  const outcome = Option.flatMap(check, check => check.outcome)
  const fields = yield* Schema.decodeUnknown(Schema.typeSchema(UpdateRequest))({
    ...options.metadata, protocol: "1", product: "desktop", channel: channel === "unknown" ? "stable" : channel,
    ts: Math.floor((yield* Clock.currentTimeMillis) / 1000), nonce: yield* newUpdateNonce,
    reason: Option.map(check, check => check.reason),
    outcome: Option.map(outcome, outcome => outcome.outcome),
    outcome_version: Option.map(outcome, outcome => outcome.version),
    outcome_reason: Option.flatMap(outcome, outcome => outcome.reason),
  }).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "request" })))
  const encoded = yield* Schema.encode(UpdateRequest)(fields).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "request" })))
  const url = yield* Effect.try({ try: () => {
    const origin = new URL(options.origin)
    if (origin.origin !== options.origin || origin.protocol !== "https:") throw new Error("Invalid update origin")
    return new URL(`${path}?${updateQuery({ ...Object.fromEntries(Object.entries(encoded).filter((entry): entry is [string, string] => typeof entry[1] === "string")), ...extra })}`, origin)
  }, catch: () => new HostedUpdateCheckFailed({ reason: "request" }) })
  const authorization = yield* options.sign(url).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "request" })))
  const http = (yield* HttpClient.HttpClient).pipe(HttpClient.withTracerDisabledWhen(() => true))
  const response = yield* http.execute(HttpClientRequest.get(url.href, {
    headers: { authorization, "user-agent": options.userAgent, accept: "application/json" },
  })).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "network" })))
  return { response, fields }
})

/** No retries or redirects. The scoped caller owns startup, hourly and explicit check admission. */
export const checkHostedUpdate = (options: HostedUpdateConnection & {
  readonly trustedPublishers: ReadonlyMap<string, KeyObject>
}, check: UpdateCheck) => Effect.gen(function* () {
  const { response, fields } = yield* signedRequest(options, "/api/update", Option.some(check))
  if (response.status === 204) return Option.none()
  if (response.status !== 200) return yield* new HostedUpdateCheckFailed({ reason: "response" })
  const bytes = yield* response.stream.pipe(Stream.runFoldEffect({ chunks: [] as Uint8Array[], size: 0 }, (state, chunk) => {
    const size = state.size + chunk.byteLength
    return size > 32 * 1024 ? new HostedUpdateCheckFailed({ reason: "response" })
      : Effect.succeed({ chunks: [...state.chunks, chunk], size })
  }), Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "response" })))
  const json = yield* Effect.try({ try: () => new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(bytes.chunks)), catch: () => new HostedUpdateCheckFailed({ reason: "response" }) })
  const offer = yield* Schema.decodeUnknown(Schema.parseJson(UpdateRelease))(json, { onExcessProperty: "error" }).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "response" })))
  const target = yield* Schema.decodeUnknown(ReleaseTarget)({ os: fields.os, arch: fields.arch, package: fields.package }).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "request" })))
  const release = yield* verifyUpdateRelease(offer, target, options.trustedPublishers).pipe(Effect.mapError(() => new HostedUpdateCheckFailed({ reason: "publisher" })))
  if (!acceptsUpdateRelease(release, fields.version)) return yield* new HostedUpdateCheckFailed({ reason: "response" })
  return Option.some(release)
}).pipe(
  Effect.provideService(FetchHttpClient.RequestInit, { redirect: "manual" }),
  Effect.timeoutFail({ duration: "10 seconds", onTimeout: () => new HostedUpdateCheckFailed({ reason: "network" }) }),
)

/** Resolve once, then transfer bytes without forwarding installation credentials to storage. */
export const resolveHostedDownload = (options: HostedUpdateConnection & {
  readonly release: UpdateRelease
}) => Effect.gen(function* () {
  const { response, fields } = yield* signedRequest(options, "/api/download", Option.none(), { release: options.release.version })
  if (!acceptsUpdateRelease(options.release, fields.version) || response.status !== 302) return yield* new HostedUpdateCheckFailed({ reason: "response" })
  return yield* Effect.try({ try: () => {
    const location = response.headers.location
    if (!location || !isGithubReleaseAssetUrl(location)) throw new Error("Unexpected artifact redirect")
    return location
  }, catch: () => new HostedUpdateCheckFailed({ reason: "response" }) })
}).pipe(
  Effect.provideService(FetchHttpClient.RequestInit, { redirect: "manual" }),
  Effect.timeoutFail({ duration: "10 seconds", onTimeout: () => new HostedUpdateCheckFailed({ reason: "network" }) }),
)
