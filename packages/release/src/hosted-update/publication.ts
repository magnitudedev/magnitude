import { Context, Effect, Schema } from "effect"
import type { KeyObject } from "node:crypto"
import { PublisherKeyId, SignedUpdateManifest, UpdateManifest, signUpdateManifest } from "./manifest"

export class ReleasePublicationFailed extends Schema.TaggedError<ReleasePublicationFailed>()("ReleasePublicationFailed", {
  stage: Schema.Literal("batch", "database", "conflict"),
}) {}
export interface ReleasePublicationStore {
  readonly promote: (envelopes: readonly SignedUpdateManifest[]) => Effect.Effect<void, ReleasePublicationFailed>
}
export const ReleasePublicationStore = Context.GenericTag<ReleasePublicationStore>("release/ReleasePublicationStore")

export const ReleasePublicationBatch = Schema.Array(UpdateManifest).pipe(
  Schema.minItems(1), Schema.maxItems(16),
  Schema.filter(batch => {
    const first = batch[0]!
    const targets = new Set<string>(), ids = new Set<string>()
    return batch.every(manifest => {
      const { os, arch, package: format } = manifest.artifact.target
      const target = `${os}/${arch}/${format}`
      if (manifest.version !== first.version || manifest.commit !== first.commit || manifest.tag !== first.tag || targets.has(target) || ids.has(manifest.artifact.id)) return false
      targets.add(target); ids.add(manifest.artifact.id)
      return true
    })
  }),
)

/** Sign only metadata already admitted from the accepted public GitHub release. */
export const prepareHostedRelease = (options: {
  readonly artifacts: typeof ReleasePublicationBatch.Type
  readonly keyId: typeof PublisherKeyId.Type
  readonly privateKey: KeyObject
}) => Effect.gen(function* () {
  const batch = yield* Schema.decodeUnknown(ReleasePublicationBatch)(options.artifacts).pipe(Effect.mapError(() => new ReleasePublicationFailed({ stage: "batch" })))
  const envelopes = yield* Effect.forEach(batch, manifest => signUpdateManifest(manifest, options.keyId, options.privateKey))
  return envelopes
})

export const publishHostedRelease = (options: Parameters<typeof prepareHostedRelease>[0]) => Effect.gen(function* () {
  const store = yield* ReleasePublicationStore
  const envelopes = yield* prepareHostedRelease(options)
  yield* store.promote(envelopes)
  return envelopes
})
