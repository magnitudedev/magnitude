import { FileSystem } from "@effect/platform"
import { ReleaseManifestSchema, validateReleaseManifest } from "@magnitudedev/release/contracts"
import { Effect, Option, Schema, Stream } from "effect"
import { createHash } from "node:crypto"
import { join, sep } from "node:path"
import { ArchiveExtractor } from "../../release/src/archive"
import { downloadObject } from "./artifact-store"
import { AssertionFailure, Digest, InfrastructureFailure, Target } from "./domain"

const Repository = Schema.String.pipe(Schema.pattern(/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/), Schema.brand("ModelRepository"))
const Revision = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{40}$/), Schema.brand("ModelRepositoryRevision"))
const ModelFilePath = Schema.NonEmptyString.pipe(Schema.filter(path => !path.includes("\\") && !path.includes(":") && !/[\x00-\x1f\x7f]/.test(path)
  && path.split("/").every(part => part !== "" && part !== "." && part !== "..")), Schema.brand("ModelFilePath"))
export const ExpectedModelFile = Schema.Struct({ repository: Repository, revision: Revision, path: ModelFilePath,
  bytes: Schema.Int.pipe(Schema.positive()), sha256: Digest })
export const ModelFileReceipt = Schema.Struct({ files: Schema.NonEmptyArray(ExpectedModelFile) })
const Package = Schema.Struct({ package: Schema.Struct({ source: Schema.TaggedStruct("HuggingFace", { repository: Repository, revision: Revision }),
  files: Schema.NonEmptyArray(Schema.Struct({ path: ModelFilePath, sizeBytes: Schema.Int.pipe(Schema.positive()), sha256: Digest })) }) })
const Manifest = Schema.Struct({ plannerInputs: Schema.Record({ key: Schema.String, value: Schema.Struct({ modelId: Schema.NonEmptyString,
  variantId: Schema.NonEmptyString, target: Package, draft: Schema.optionalWith(Package, { as: "Option", exact: true }) }) }) })
const fail = (message: string) => new AssertionFailure({ message })

/** Read only the bounded catalog declaration; archive admission authenticates the containing bundle bytes. */
export const declaredModelFiles = (bundle: Uint8Array, model: string) => Effect.gen(function* () {
  const bytes = Buffer.from(bundle.buffer, bundle.byteOffset, bundle.byteLength)
  if (bytes.length < 20 || bytes.subarray(0, 8).toString() !== "MAGPLAN3") return yield* fail("Admitted model catalog has an invalid bundle header")
  const length = bytes.readBigUInt64LE(8)
  if (length === 0n || length > 64n * 1024n * 1024n || length > BigInt(bytes.length - 20)) return yield* fail("Admitted model catalog manifest exceeds its bound")
  const manifest = yield* Schema.decodeUnknown(Schema.parseJson(Manifest))(bytes.subarray(16, 16 + Number(length)).toString("utf8")).pipe(
    Effect.mapError(() => fail("Admitted model catalog has an invalid file declaration")))
  const selected = Object.values(manifest.plannerInputs).filter(input => `${input.modelId}:${input.variantId}` === model)
  if (selected.length !== 1) return yield* fail("Test model does not uniquely match the admitted catalog")
  const input = selected[0]!
  const files: (typeof ExpectedModelFile.Type)[] = []
  for (const entry of [input.target, ...Option.toArray(input.draft)]) for (const file of entry.package.files) {
    const expected = ExpectedModelFile.make({ repository: entry.package.source.repository, revision: entry.package.source.revision, path: file.path, bytes: file.sizeBytes, sha256: file.sha256 })
    const previous = files.find(item => item.repository === expected.repository && item.revision === expected.revision && item.path === expected.path)
    if (previous && (previous.sha256 !== expected.sha256 || previous.bytes !== expected.bytes)) return yield* fail("Model catalog contains conflicting file identities")
    if (!previous) files.push(expected)
  }
  return yield* Schema.decodeUnknown(ModelFileReceipt)({ files })
})

export const admittedModelFiles = (release: typeof ReleaseManifestSchema.Type, host: Target["artifactHost"], model: string) => Effect.scoped(Effect.gen(function* () {
  yield* validateReleaseManifest(release)
  const fs = yield* FileSystem.FileSystem, extractor = yield* ArchiveExtractor
  const roots = release.artifacts.filter(artifact => artifact.kind === "icn-base" && Option.contains(artifact.host, host))
  if (roots.length !== 1) return yield* fail("Model integrity verification requires one admitted native base")
  const artifact = roots[0]!, root = yield* fs.makeTempDirectoryScoped({ prefix: "ml-model-files-" })
  const archive = join(root, "base.tar.gz"), destination = join(root, "runtime")
  yield* downloadObject(Digest.make(artifact.sha256), archive)
  if (Number((yield* fs.stat(archive)).size) !== artifact.bytes) return yield* fail("Native archive length differs from admitted metadata")
  yield* extractor.extract(archive, destination, artifact, Option.none())
  const path = join(destination, "catalog", "model-planner-inputs.bundle")
  if (Number((yield* fs.stat(path)).size) > 256 * 1024 * 1024) return yield* fail("Admitted catalog bundle exceeds inspection bound")
  return yield* declaredModelFiles(yield* fs.readFile(path), model)
})).pipe(Effect.mapError(error => error._tag === "AssertionFailure" ? error : new InfrastructureFailure({ operation: "model-files", message: error.message })))

/** Independently stream-hash every target/companion file, never export model contents. */
export const verifyModelFiles = (profile: string, expected: typeof ModelFileReceipt.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = join(yield* fs.realPath(profile), "models")
  if ((yield* fs.realPath(root)) !== root) return yield* fail("Managed model root is not the owned physical directory")
  for (const file of expected.files) {
    const repository = join(root, "hub", `models--${file.repository.replaceAll("/", "--")}`)
    const snapshot = join(repository, "snapshots", file.revision), path = join(snapshot, file.path)
    if ((yield* fs.realPath(repository)) !== repository || (yield* fs.realPath(snapshot)) !== snapshot) return yield* fail("Model snapshot directory escapes its owned repository")
    const physical = yield* fs.realPath(path)
    if (!physical.startsWith(repository + sep)) return yield* fail("Model snapshot file escapes its owned repository")
    const before = yield* fs.stat(path)
    if (before.type !== "File" || Number(before.size) !== file.bytes) return yield* fail(`Model file size differs from admitted catalog: ${file.path}`)
    const hash = createHash("sha256")
    let size = 0
    yield* fs.stream(path).pipe(Stream.runForEach(chunk => Effect.gen(function* () {
      size += chunk.byteLength
      if (size > file.bytes) return yield* fail(`Model file grew beyond its declared size: ${file.path}`)
      hash.update(chunk)
    })))
    const after = yield* fs.stat(path)
    if (size !== file.bytes || Number(after.size) !== file.bytes || before.dev !== after.dev || !Option.getEquivalence<number>((a, b) => a === b)(before.ino, after.ino)
      || (yield* fs.realPath(path)) !== physical || hash.digest("hex") !== file.sha256) return yield* fail(`Model bytes differ from admitted catalog: ${file.path}`)
  }
  return expected
}).pipe(Effect.mapError(error => error._tag === "AssertionFailure" ? error : error._tag === "SystemError" && error.reason === "NotFound"
  ? fail("Installed model is missing declared files") : new InfrastructureFailure({ operation: "model-files", message: error.message })))
