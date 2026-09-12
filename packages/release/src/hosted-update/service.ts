import { Clock, Context, Effect, Option, Schema } from "effect"
import type { KeyObject } from "node:crypto"
import { decodeUpdateRequest, UpdateRequest } from "./request"
import { verifyUpdateRequest, InstallationId, type RequestNonce } from "./request-auth"
import { isNewerVersion } from "../client-update/release-channels"
import { acceptsUpdateManifest, verifyUpdateManifest, SignedUpdateManifest, ArtifactId, ArtifactTarget } from "./manifest"

export class DistributionStoreUnavailable extends Schema.TaggedError<DistributionStoreUnavailable>()("DistributionStoreUnavailable", {}) {}
export const Country = Schema.String.pipe(Schema.pattern(/^[A-Z]{2}$/))
export const CheckObservation = Schema.Struct({
  installation: InstallationId,
  request: UpdateRequest,
  country: Schema.optionalWith(Country, { as: "Option", exact: true }),
  offeredVersion: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})
export const DownloadObservation = Schema.Struct({
  installation: InstallationId,
  request: UpdateRequest,
  country: Schema.optionalWith(Country, { as: "Option", exact: true }),
  release: Schema.String,
  artifact: ArtifactId,
})
export const InstallerDownloadObservation = Schema.Struct({
  target: ArtifactTarget,
  country: Schema.optionalWith(Country, { as: "Option", exact: true }),
  release: Schema.String,
  artifact: ArtifactId,
})
export interface DistributionStore {
  /** Atomic unique admission; expiry is server time, not an untrusted client TTL. */
  readonly admit: (installation: InstallationId, nonce: RequestNonce, expiresAt: number) => Effect.Effect<boolean, DistributionStoreUnavailable>
  readonly candidates: (target: Pick<UpdateRequest, "os" | "arch"> & { readonly package: UpdateRequest["package"] | "dmg" }) => Effect.Effect<readonly SignedUpdateManifest[], DistributionStoreUnavailable>
  readonly recordCheck: (observation: typeof CheckObservation.Type) => Effect.Effect<void, DistributionStoreUnavailable>
  readonly artifact: (release: string, id: ArtifactId) => Effect.Effect<Option.Option<SignedUpdateManifest>, DistributionStoreUnavailable>
  readonly recordDownload: (observation: typeof DownloadObservation.Type) => Effect.Effect<void, DistributionStoreUnavailable>
  readonly recordInstallerDownload: (observation: typeof InstallerDownloadObservation.Type) => Effect.Effect<void, DistributionStoreUnavailable>
}
export const DistributionStore = Context.GenericTag<DistributionStore>("release/DistributionStore")

const headers = { "Cache-Control": "private, no-store", "Content-Type": "application/json", "X-Content-Type-Options": "nosniff" }
const response = (status: number, body: string | null = null) => new Response(body, { status, headers })

/** HTTP boundary. Country is supplied by trusted host composition, never read from caller headers here. */
export const handleUpdateCheck = (request: Request, options: {
  readonly origin: string
  readonly country: Option.Option<string>
  readonly trustedPublishers: ReadonlyMap<string, KeyObject>
}) => Effect.gen(function* () {
  const url = new URL(request.url)
  if (url.origin !== options.origin || url.pathname !== "/api/update") return response(404)
  if (request.method !== "GET") return response(405)
  const now = Math.floor((yield* Clock.currentTimeMillis) / 1000)
  const fields = yield* decodeUpdateRequest(url, now)
  const installation = yield* verifyUpdateRequest(request.headers.get("authorization") ?? "", url)
  const store = yield* DistributionStore
  if (!(yield* store.admit(installation, fields.nonce, now + 600))) return response(409)
  const candidates = yield* store.candidates(fields)
  let offer: Option.Option<SignedUpdateManifest> = Option.none()
  let offeredVersion: Option.Option<string> = Option.none()
  for (const candidate of candidates) {
    const manifest = yield* verifyUpdateManifest(candidate, options.trustedPublishers)
    if (acceptsUpdateManifest(manifest, fields) && (Option.isNone(offeredVersion) || isNewerVersion(manifest.version, offeredVersion.value))) {
      offer = Option.some(candidate)
      offeredVersion = Option.some(manifest.version)
    }
  }
  const country = Option.filter(options.country, Schema.is(Country))
  yield* store.recordCheck({ installation, request: fields, country, offeredVersion }).pipe(
    Effect.catchTag("DistributionStoreUnavailable", () => Effect.logWarning("Update telemetry write unavailable")),
  )
  if (Option.isNone(offer)) return response(204)
  const body = yield* Schema.encode(Schema.parseJson(SignedUpdateManifest))(offer.value).pipe(Effect.mapError(() => new DistributionStoreUnavailable()))
  return response(200, body)
}).pipe(Effect.catchTags({
  InvalidUpdateRequest: () => Effect.succeed(response(400)),
  ExpiredUpdateRequest: () => Effect.succeed(response(401)),
  InvalidUpdateSignature: () => Effect.succeed(response(401)),
  InvalidUpdateManifest: () => Effect.succeed(response(503)),
  DistributionStoreUnavailable: () => Effect.succeed(response(503)),
}))
