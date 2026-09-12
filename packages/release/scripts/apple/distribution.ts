import * as FileSystem from "@effect/platform/FileSystem"
import { Config, Effect, Option, Schema } from "effect"
import { basename, resolve } from "node:path"
import { ArtifactDigest, SourceCommit, sha256File } from "../../src/macos-app"
import { type ReleaseArtifact } from "../../src/contracts"
import { AppleDistributionFailed, appleCommand, appleSigning, notarize } from "./signing"
import { lstat } from "node:fs/promises"

export const AppleNotaryAcceptance = Schema.Struct({
  id: Schema.String.pipe(Schema.minLength(1)),
  status: Schema.Literal("Accepted"),
  inputSha256: ArtifactDigest,
  unit: Schema.String,
})
export const AppleDistributionReceipt = Schema.Struct({
  sourceCommit: SourceCommit,
  team: Schema.String.pipe(Schema.pattern(/^[A-Z0-9]{10}$/)),
  artifacts: Schema.NonEmptyArray(Schema.Struct({ id: Schema.String, sha256: ArtifactDigest })),
  notarizations: Schema.NonEmptyArray(AppleNotaryAcceptance),
  stapledApp: Schema.Boolean,
})

export const AppleConsumerReceipt = Schema.Struct({
  sourceCommit: SourceCommit,
  artifacts: Schema.NonEmptyArray(Schema.Struct({ id: Schema.String, sha256: ArtifactDigest })),
})

export const writeAppleConsumerReceipt = (output: string, artifacts: readonly ReleaseArtifact[]) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const receipt = yield* Schema.decodeUnknown(AppleConsumerReceipt)({
    sourceCommit: yield* Config.string("MAGNITUDE_SOURCE_COMMIT"), artifacts: artifacts.map(({ id, sha256 }) => ({ id, sha256 })),
  })
  yield* fs.writeFileString(resolve(output, "apple-consumer.receipt.json"), yield* Schema.encode(Schema.parseJson(AppleConsumerReceipt))(receipt))
})

export const regularAppleFiles = (directory: string): Effect.Effect<readonly { path: string; source: string; mode: number }[], import("@effect/platform/Error").PlatformError | AppleDistributionFailed, FileSystem.FileSystem> => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const files: { path: string; source: string; mode: number }[] = []
  const visit = (relative: string): Effect.Effect<void, import("@effect/platform/Error").PlatformError | AppleDistributionFailed> => Effect.gen(function* () {
    const absolute = resolve(directory, relative)
    for (const name of (yield* fs.readDirectory(absolute)).sort()) {
      const path = relative ? `${relative}/${name}` : name
      const source = resolve(directory, path)
      // lstat: links and special files cannot enter the regular-file release format.
      const linked = yield* Effect.tryPromise({ try: () => lstat(source), catch: (error) => new AppleDistributionFailed({ message: `Cannot inspect ${source}: ${String(error)}` }) })
      if (linked.isSymbolicLink()) return yield* new AppleDistributionFailed({ message: `Apple release archives cannot contain symlinks: ${source}` })
      const info = yield* fs.stat(source)
      if (info.type === "Directory") yield* visit(path)
      else if (info.type === "File") files.push({ path, source, mode: info.mode & 0o777 })
      else return yield* new AppleDistributionFailed({ message: `Unsupported Apple archive file ${source}` })
    }
  })
  yield* visit("")
  return files
})

/** Expose embedded native inputs alongside compiled executables for Apple's scanner. */
export const notarizeAppleUnit = (unit: string, output: string, sources: readonly string[]) => Effect.gen(function* () {
  const signing = yield* appleSigning
  if (signing.mode === "adhoc") return Option.none<typeof AppleNotaryAcceptance.Type>()
  const fs = yield* FileSystem.FileSystem
  const directory = resolve(output, ".apple", unit)
  yield* fs.makeDirectory(directory, { recursive: true })
  for (const [index, source] of sources.entries()) {
    const destination = resolve(directory, `${index}-${basename(source)}`)
    if (yield* fs.exists(destination)) return yield* new AppleDistributionFailed({ message: `Notarization staging already exists: ${destination}` })
    // Preserve sealed framework links verbatim; generic recursive copies may rewrite them.
    yield* appleCommand("/usr/bin/ditto", source, destination)
  }
  const archive = resolve(output, ".apple", `${unit}.zip`)
  yield* appleCommand("/usr/bin/ditto", "-c", "-k", "--keepParent", directory, archive)
  const digest = yield* sha256File(archive)
  const result = yield* notarize(archive, resolve(output, ".apple", `${unit}.notary.json`))
  if (result === undefined) return yield* Effect.dieMessage("Production notarization returned no acceptance")
  return Option.some(yield* Schema.decodeUnknown(AppleNotaryAcceptance)({ ...result, inputSha256: digest, unit }))
})

export const writeAppleReceipt = (output: string, artifacts: readonly ReleaseArtifact[], notarizations: readonly Option.Option<typeof AppleNotaryAcceptance.Type>[], stapledApp: boolean) => Effect.gen(function* () {
  const signing = yield* appleSigning
  if (signing.mode === "adhoc") return
  const fs = yield* FileSystem.FileSystem
  const receipt = yield* Schema.decodeUnknown(AppleDistributionReceipt)({
    sourceCommit: yield* Config.string("MAGNITUDE_SOURCE_COMMIT"), team: signing.team,
    artifacts: artifacts.map(({ id, sha256 }) => ({ id, sha256 })),
    notarizations: notarizations.map(Option.getOrThrow), stapledApp,
  })
  yield* fs.writeFileString(resolve(output, "apple-distribution.receipt.json"), yield* Schema.encode(Schema.parseJson(AppleDistributionReceipt))(receipt), { mode: 0o600 })
})
