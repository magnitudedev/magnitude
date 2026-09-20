import { Context, Effect, Layer, Option, Schema, Stream } from "effect"
import { ReleaseManifestSchema, validateReleaseManifest } from "@magnitudedev/release/contracts"
import { ArtifactStore } from "./artifact-store"
import { Database, decodeRow } from "./database"
import { Digest, InfrastructureFailure, Input, InvalidInput, OwnerId } from "./domain"
import { isNewerVersion } from "@magnitudedev/release"
import { decodeUpdateConfiguration, UpdateConfiguration } from "@magnitudedev/release/hosted-update"
import { UpdateAcceptance } from "./update-acceptance"
import { SourceManifest, validateManifest } from "./snapshot"

export const ArtifactInput = Schema.Struct({ schemaVersion: Schema.Literal(1), kind: Schema.Literal("artifacts"), release: ReleaseManifestSchema,
  updateAcceptance: Schema.optionalWith(UpdateAcceptance, { as: "Option", exact: true }) })
export const artifactReleases = (input: typeof ArtifactInput.Type) => [input.release, ...Option.match(input.updateAcceptance, { onNone: () => [], onSome: pair => [pair.previous, pair.candidate] })]
export const artifactObjects = (input: typeof ArtifactInput.Type) => [...artifactReleases(input).flatMap(release => release.artifacts.map(artifact => ({ sha256: Digest.make(artifact.sha256), bytes: artifact.bytes }))),
  ...Option.match(input.updateAcceptance, { onNone: () => [], onSome: pair => [pair.authority] })]
export const InputManifest = Schema.Union(SourceManifest, ArtifactInput)
export class InputDenied extends Schema.TaggedError<InputDenied>()("InputDenied", {}) {}
type Owner = typeof OwnerId.Type
type InputRef = typeof Input.Type
export interface InputRegistry {
  readonly missing: (owner: Owner, digests: readonly Digest[]) => Effect.Effect<readonly Digest[], InfrastructureFailure>
  readonly upload: (owner: Owner, digest: Digest, bytes: Stream.Stream<Uint8Array, InfrastructureFailure>) => Effect.Effect<void, InfrastructureFailure>
  readonly read: (owner: Owner, digest: Digest) => Effect.Effect<Stream.Stream<Uint8Array, InfrastructureFailure>, InfrastructureFailure | InputDenied>
  readonly register: (owner: Owner, input: InputRef) => Effect.Effect<void, InfrastructureFailure | InvalidInput | InputDenied>
  readonly require: (owner: Owner, input: InputRef) => Effect.Effect<void, InfrastructureFailure | InputDenied>
}
export const InputRegistry = Context.GenericTag<InputRegistry>("@magnitudedev/testing-lab/InputRegistry")
export const InputRegistryLive = Layer.effect(InputRegistry, Effect.gen(function* () {
  const db = yield* Database
  const objects = yield* ArtifactStore
  const allowed = (owner: Owner, digest: Digest) => Effect.gen(function* () {
    const rows = yield* db.query("SELECT bytes FROM lab_objects WHERE owner=$1 AND digest=$2", [owner, digest])
    if (!rows[0]) return yield* new InputDenied({})
    return yield* decodeRow(Schema.Struct({ bytes: Schema.NumberFromString }), rows[0])
  })
  return {
    missing: (owner, digests) => Effect.gen(function* () {
      const rows = yield* db.query("SELECT digest FROM lab_objects WHERE owner=$1 AND digest = ANY($2::text[])", [owner, digests])
      const owned = new Set(yield* Effect.forEach(rows, row => decodeRow(Schema.Struct({ digest: Digest }), row).pipe(Effect.map(row => row.digest))))
      return [...new Set(digests)].filter(digest => !owned.has(digest))
    }),
    upload: (owner, digest, content) => Effect.gen(function* () {
      let bytes = 0
      yield* objects.put(digest, content.pipe(Stream.tap(chunk => Effect.sync(() => { bytes += chunk.byteLength }))))
      yield* db.query(`INSERT INTO lab_objects(owner,digest,bytes) VALUES($1,$2,$3)
        ON CONFLICT(owner,digest) DO NOTHING`, [owner, digest, bytes])
    }),
    read: (owner, digest) => allowed(owner, digest).pipe(Effect.as(objects.get(digest))),
    register: (owner, input) => Effect.gen(function* () {
      const meta = yield* allowed(owner, input.digest)
      if (meta.bytes > 16 * 1024 * 1024) return yield* new InvalidInput({ message: "Input manifest exceeds 16 MiB" })
      let manifestBytes = 0
      const chunks = yield* objects.get(input.digest).pipe(Stream.tap(chunk => Effect.gen(function* () {
        manifestBytes += chunk.byteLength
        if (manifestBytes > meta.bytes || manifestBytes > 16 * 1024 * 1024) return yield* new InvalidInput({ message: "Input manifest exceeds its admitted length" })
      })), Stream.runCollect)
      const json = Buffer.concat(Array.from(chunks)).toString("utf8")
      const manifest = yield* Schema.decodeUnknown(Schema.parseJson(InputManifest))(json).pipe(Effect.mapError(() => new InvalidInput({ message: "Malformed input manifest" })))
      if (manifest.kind !== input.kind) return yield* new InvalidInput({ message: "Input kind does not match its manifest" })
      if (manifest.kind === "source") yield* validateManifest(manifest)
      else {
        for (const release of artifactReleases(manifest)) yield* validateReleaseManifest(release).pipe(Effect.mapError(error => new InvalidInput({ message: error.message })))
        if (Option.isSome(manifest.updateAcceptance)) {
          const pair = manifest.updateAcceptance.value
          if (pair.previous.sourceCommit !== manifest.release.sourceCommit || pair.candidate.sourceCommit !== manifest.release.sourceCommit || !isNewerVersion(pair.candidate.version, pair.previous.version)) {
            return yield* new InvalidInput({ message: "Update acceptance pair differs from its source or version order" })
          }
          yield* Schema.encode(UpdateConfiguration)(pair.configuration).pipe(Effect.flatMap(wire => decodeUpdateConfiguration(wire, true)), Effect.mapError(() => new InvalidInput({ message: "Invalid update acceptance publisher configuration" })))
        }
        if (artifactReleases(manifest).flatMap(release => release.artifacts).some(a => a.filename.includes("/") || a.filename.includes("\\") || a.filename === "." || a.filename === "..")) {
          return yield* new InvalidInput({ message: "Artifact filenames must be basenames" })
        }
      }
      const files = manifest.kind === "source" ? manifest.entries.filter(e => e.kind === "file") : artifactObjects(manifest)
      const digests = [...new Set(files.map(file => file.sha256))]
      const lengths = new Map<string, number>()
      for (let offset = 0; offset < digests.length; offset += 1000) {
        const rows = yield* db.query("SELECT digest,bytes FROM lab_objects WHERE owner=$1 AND digest=ANY($2::text[])", [owner, digests.slice(offset, offset + 1000)])
        const uploaded = yield* decodeRow(Schema.Array(Schema.Struct({ digest: Digest, bytes: Schema.NumberFromString })), rows)
        for (const file of uploaded) lengths.set(file.digest, file.bytes)
      }
      for (const file of files) {
        const uploaded = lengths.get(file.sha256)
        if (uploaded === undefined) return yield* new InputDenied({})
        if (uploaded !== file.bytes) return yield* new InvalidInput({ message: "Manifest object byte count does not match uploaded data" })
      }
      yield* db.query("INSERT INTO lab_inputs(owner,digest,kind) VALUES($1,$2,$3) ON CONFLICT(owner,digest) DO NOTHING", [owner, input.digest, input.kind])
    }),
    require: (owner, input) => Effect.gen(function* () {
      const rows = yield* db.query("SELECT 1 FROM lab_inputs WHERE owner=$1 AND digest=$2 AND kind=$3", [owner, input.digest, input.kind])
      if (!rows.length) return yield* new InputDenied({})
    }),
  } satisfies InputRegistry
}))
