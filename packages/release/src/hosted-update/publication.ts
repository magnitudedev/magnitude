import { Context, Effect, Schema } from "effect"
import type { KeyObject } from "node:crypto"
import { PublisherKeyId, SignedUpdateManifest, UpdateManifest, signUpdateManifest } from "./manifest"
import { publishArtifactBytes } from "./blob-publication"

export class ReleasePublicationFailed extends Schema.TaggedError<ReleasePublicationFailed>()("ReleasePublicationFailed", {
  stage: Schema.Literal("batch", "database", "conflict"),
}) {}
export interface ReleasePublicationStore {
  readonly promote: (envelopes: readonly SignedUpdateManifest[]) => Effect.Effect<void, ReleasePublicationFailed>
}
export const ReleasePublicationStore = Context.GenericTag<ReleasePublicationStore>("release/ReleasePublicationStore")

export const ReleasePublicationBatch = Schema.Array(Schema.Struct({ file: Schema.String, manifest: UpdateManifest })).pipe(
  Schema.minItems(1), Schema.maxItems(16),
  Schema.filter(batch => {
    const first = batch[0]!.manifest
    const targets = new Set<string>(), ids = new Set<string>()
    return batch.every(({ manifest }) => {
      const { os, arch, package: format } = manifest.artifact.target
      const target = `${os}/${arch}/${format}`
      if (manifest.version !== first.version || manifest.commit !== first.commit || targets.has(target) || ids.has(manifest.artifact.id)) return false
      targets.add(target); ids.add(manifest.artifact.id)
      return true
    })
  }),
)

/** No channel moves until every local artifact and full remote transfer has been verified. */
export const publishHostedRelease = (options: {
  readonly artifacts: typeof ReleasePublicationBatch.Type
  readonly keyId: typeof PublisherKeyId.Type
  readonly privateKey: KeyObject
  readonly token: string
  readonly storageOrigin: string
}) => Effect.gen(function* () {
  const batch = yield* Schema.decodeUnknown(ReleasePublicationBatch)(options.artifacts).pipe(Effect.mapError(() => new ReleasePublicationFailed({ stage: "batch" })))
  const store = yield* ReleasePublicationStore
  const envelopes = yield* Effect.forEach(batch, artifact => Effect.gen(function* () {
    yield* publishArtifactBytes({ ...artifact, token: options.token, storageOrigin: options.storageOrigin })
    return yield* signUpdateManifest(artifact.manifest, options.keyId, options.privateKey)
  }), { concurrency: 2 })
  yield* store.promote(envelopes)
  return envelopes
})
