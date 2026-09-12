import { Effect, Option, Schema } from "effect"
import type { KeyObject } from "node:crypto"
import { ArtifactTarget, verifyUpdateManifest, type UpdateManifest } from "./manifest"
import { Country, DistributionStore } from "./service"
import { isNewerVersion, releaseChannelOf } from "../client-update/release-channels"

const InstallerTarget = ArtifactTarget.pipe(Schema.filter(target => target.package !== "mac-zip"))
const response = (status: number) => new Response(null, { status, headers: { "Cache-Control": "private, no-store" } })

/** First downloads have no installation identity. Count requests, never pretend they identify users. */
export const handleInstallerDownload = (request: Request, options: {
  readonly origin: string
  readonly storageOrigin: string
  readonly country: Option.Option<string>
  readonly trustedPublishers: ReadonlyMap<string, KeyObject>
}) => Effect.gen(function* () {
  const url = new URL(request.url)
  if (url.origin !== options.origin || url.pathname !== "/api/installer") return response(404)
  if (request.method !== "GET") return response(405)
  if ([...url.searchParams.keys()].some(key => url.searchParams.getAll(key).length !== 1)) return response(400)
  const target = yield* Schema.decodeUnknown(InstallerTarget)(Object.fromEntries(url.searchParams), { onExcessProperty: "error" })
  const store = yield* DistributionStore
  const candidates = yield* store.candidates(target)
  let selected: Option.Option<UpdateManifest> = Option.none()
  for (const envelope of candidates) {
    const manifest = yield* verifyUpdateManifest(envelope, options.trustedPublishers)
    if (releaseChannelOf(manifest.version) !== "stable" || manifest.artifact.target.os !== target.os || manifest.artifact.target.arch !== target.arch
      || manifest.artifact.target.package !== target.package) continue
    if (Option.isNone(selected) || isNewerVersion(manifest.version, selected.value.version)) selected = Option.some(manifest)
  }
  if (Option.isNone(selected)) return response(404)
  const manifest = selected.value
  const location = new URL(manifest.artifact.path, options.storageOrigin + "/")
  if (location.protocol !== "https:" || location.origin !== options.storageOrigin) return response(503)
  yield* store.recordInstallerDownload({ target, release: manifest.version, artifact: manifest.artifact.id, country: Option.filter(options.country, Schema.is(Country)) }).pipe(
    Effect.catchTag("DistributionStoreUnavailable", () => Effect.logWarning("Installer download telemetry write unavailable")),
  )
  return new Response(null, { status: 302, headers: { "Cache-Control": "private, no-store", Location: location.href } })
}).pipe(Effect.catchTags({
  ParseError: () => Effect.succeed(response(400)),
  InvalidUpdateManifest: () => Effect.succeed(response(503)),
  DistributionStoreUnavailable: () => Effect.succeed(response(503)),
}))
