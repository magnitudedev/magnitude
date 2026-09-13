import { Clock, Effect, Option, Schema } from "effect"
import type { KeyObject } from "node:crypto"
import { decodeUpdateRequest } from "./request"
import { verifyUpdateRequest } from "./request-auth"
import { ArtifactId, verifyUpdateManifest } from "./manifest"
import { Country, DistributionStore } from "./service"
import { isValidVersion } from "../client-update/release-channels"

const response = (status: number) => new Response(null, { status, headers: { "Cache-Control": "private, no-store" } })

/** A redirect records intent once; range requests go directly to object storage without installation credentials. */
export const handleArtifactDownload = (request: Request, options: {
  readonly origin: string
  readonly storageOrigin: string
  readonly country: Option.Option<string>
  readonly trustedPublishers: ReadonlyMap<string, KeyObject>
}) => Effect.gen(function* () {
  const url = new URL(request.url)
  if (url.origin !== options.origin || url.pathname !== "/api/download") return response(404)
  if (request.method !== "GET") return response(405)
  const releases = url.searchParams.getAll("release")
  const release = releases[0]
  if (releases.length !== 1 || !release || release.length > 96 || !isValidVersion(release)) return response(400)
  const artifacts = url.searchParams.getAll("artifact")
  if (artifacts.length !== 1 || !Schema.is(ArtifactId)(artifacts[0])) return response(400)
  const artifact = ArtifactId.make(artifacts[0])
  const fieldsUrl = new URL(url)
  fieldsUrl.searchParams.delete("release")
  fieldsUrl.searchParams.delete("artifact")
  const now = Math.floor((yield* Clock.currentTimeMillis) / 1000)
  const fields = yield* decodeUpdateRequest(fieldsUrl, now)
  const installation = yield* verifyUpdateRequest(request.headers.get("authorization") ?? "", url)
  const store = yield* DistributionStore
  if (!(yield* store.admit(installation, fields.nonce, now + 600))) return response(409)
  const envelope = yield* store.artifact(release, artifact)
  if (Option.isNone(envelope)) return response(404)
  const manifest = yield* verifyUpdateManifest(envelope.value, options.trustedPublishers)
  if (manifest.version !== release || manifest.artifact.id !== artifact || manifest.artifact.target.os !== fields.os
    || manifest.artifact.target.arch !== fields.arch || manifest.artifact.target.package !== fields.package) return response(404)
  yield* store.recordDownload({ installation, request: fields, country: Option.filter(options.country, Schema.is(Country)), release, artifact }).pipe(
    Effect.catchTag("DistributionStoreUnavailable", () => Effect.logWarning("Download telemetry write unavailable")),
  )
  const location = new URL(manifest.artifact.path, options.storageOrigin + "/")
  if (location.protocol !== "https:" || location.origin !== options.storageOrigin) return response(503)
  return new Response(null, { status: 302, headers: { "Cache-Control": "private, no-store", Location: location.href } })
}).pipe(Effect.catchTags({
  InvalidUpdateRequest: () => Effect.succeed(response(400)),
  ExpiredUpdateRequest: () => Effect.succeed(response(401)),
  InvalidUpdateSignature: () => Effect.succeed(response(401)),
  InvalidUpdateManifest: () => Effect.succeed(response(503)),
  DistributionStoreUnavailable: () => Effect.succeed(response(503)),
}))
