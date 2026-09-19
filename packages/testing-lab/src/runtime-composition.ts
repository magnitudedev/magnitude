import { FileSystem } from "@effect/platform"
import { ReleaseManifestSchema, validateReleaseManifest } from "@magnitudedev/release/contracts"
import { Effect, Option } from "effect"
import { join } from "node:path"
import { ArchiveExtractor } from "../../release/src/archive"
import { downloadObject } from "./artifact-store"
import { AssertionFailure, Digest, Target } from "./domain"

const invalid = (message: string) => new AssertionFailure({ message: `Runtime dependency composition: ${message}` })

/** Scoped inspection fixture using only verified admitted bytes and the release layout. */
export const admittedRuntimeComposition = (release: typeof ReleaseManifestSchema.Type, target: Target) => Effect.gen(function* () {
  yield* validateReleaseManifest(release)
  const fs = yield* FileSystem.FileSystem
  const extractor = yield* ArchiveExtractor
  const bases = release.artifacts.filter(artifact => artifact.kind === "icn-base" && Option.contains(artifact.host, target.artifactHost))
  const packs = target.backend === "cpu" ? [] : release.artifacts.filter(artifact => artifact.kind === "icn-backend"
    && Option.contains(artifact.host, target.artifactHost) && Option.contains(artifact.backend, target.backend))
  if (bases.length !== 1 || (target.backend !== "cpu" && packs.length !== 1)) return yield* invalid("exactly one base and the selected backend pack are required")
  const base = bases[0]!
  for (const pack of packs) {
    if (!Option.contains(pack.requiredBaseId, base.id) || !Option.contains(pack.nativeBuild, Option.getOrThrow(base.nativeBuild))
      || !Option.contains(pack.backendModuleAbi, Option.getOrThrow(base.backendModuleAbi))) return yield* invalid("backend pack does not match the selected base identity")
  }
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "ml-runtime-closure-" })
  const root = join(temporary, "composed")
  yield* fs.makeDirectory(root)
  const artifacts = [base, ...packs]
  for (const [index, artifact] of artifacts.entries()) {
    const archive = join(temporary, `${index}.tar.gz`), extracted = join(temporary, `archive-${index}`)
    yield* downloadObject(Digest.make(artifact.sha256), archive)
    if (Number((yield* fs.stat(archive)).size) !== artifact.bytes) return yield* invalid(`archive length differs from admission: ${artifact.id}`)
    yield* extractor.extract(archive, extracted, artifact, Option.none())
    for (const directory of artifact.kind === "icn-base" ? ["bin", "catalog", "runtime", "backends"] : ["runtime", "backends"]) {
      const destination = join(root, directory), source = join(extracted, directory)
      yield* fs.makeDirectory(destination, { recursive: true })
      if (!(yield* fs.exists(source))) {
        if (directory !== "runtime") return yield* invalid(`missing ${directory} in ${artifact.id}`)
        continue
      }
      for (const name of yield* fs.readDirectory(source)) {
        if (yield* fs.exists(join(destination, name))) return yield* invalid(`colliding archive path ${directory}/${name}`)
        yield* fs.copy(join(source, name), join(destination, name), { overwrite: false })
      }
    }
  }
  return { root, artifacts: artifacts.map(artifact => artifact.id) }
})
