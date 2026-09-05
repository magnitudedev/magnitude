import { Data, Effect, Option, Schema } from "effect"
import type { Backend } from "./targets"
import { PluginArtifactSchema, PluginHostSchema, RpcReleaseSchema, type PluginHost } from "./plugins"

export const CLI_PACKAGE_NAME = "@magnitudedev/cli"

export const releaseTag = (version: string): string =>
  `${CLI_PACKAGE_NAME}@${version}`

const NonEmpty = Schema.String.pipe(Schema.minLength(1))
const Sha256 = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/))
const PositiveInt = Schema.Int.pipe(Schema.greaterThan(0))
const Host = Schema.Literal(
  "darwin-arm64",
  "darwin-x64",
  "linux-arm64-gnu",
  "linux-x64-gnu",
  "windows-x64-msvc",
)
const BackendSchema = Schema.Literal("cpu", "metal", "cuda", "vulkan")

const CudaPtxImage = Schema.Struct({
  ptxVersion: Schema.String.pipe(Schema.pattern(/^\d+\.\d+$/)),
  target: PositiveInt,
  architectureSpecific: Schema.Boolean,
  minimumDriverApi: PositiveInt,
})

const Compatibility = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("metal") }),
  Schema.Struct({
    kind: Schema.Literal("cuda"),
    toolkitVersion: NonEmpty,
    compiler: NonEmpty,
    images: Schema.NonEmptyArray(CudaPtxImage),
  }),
  Schema.Struct({
    kind: Schema.Literal("vulkan"),
    minimumApi: NonEmpty,
  }),
)

export const ReleaseArtifactSchema = Schema.Struct({
  id: NonEmpty,
  kind: Schema.Literal("cli", "acn", "icn-base", "icn-backend"),
  host: Schema.optionalWith(Host, { as: "Option", exact: true }),
  backend: Schema.optionalWith(BackendSchema, { as: "Option", exact: true }),
  filename: NonEmpty,
  bytes: PositiveInt,
  sha256: Sha256,
  requiredBaseId: Schema.optionalWith(NonEmpty, { as: "Option", exact: true }),
  nativeBuild: Schema.optionalWith(NonEmpty, { as: "Option", exact: true }),
  backendModuleAbi: Schema.optionalWith(NonEmpty, { as: "Option", exact: true }),
  compatibility: Schema.optionalWith(Compatibility, { as: "Option", exact: true }),
})
export type ReleaseArtifact = typeof ReleaseArtifactSchema.Type

export const ReleaseManifestSchema = Schema.Struct({
  schemaVersion: Schema.Literal(2),
  version: NonEmpty,
  acnRevision: PositiveInt.pipe(Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER)),
  tag: NonEmpty,
  sourceCommit: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{40}$/)),
  rpc: RpcReleaseSchema,
  plugins: Schema.NonEmptyArray(PluginArtifactSchema),
  artifacts: Schema.NonEmptyArray(ReleaseArtifactSchema),
})
export type ReleaseManifest = typeof ReleaseManifestSchema.Type

export class InvalidReleaseManifest extends Data.TaggedError("InvalidReleaseManifest")<{
  readonly message: string
}> {}

export const decodeReleaseManifest = (bytes: Uint8Array) =>
  Schema.decodeUnknown(Schema.parseJson(ReleaseManifestSchema))(
    new TextDecoder().decode(bytes),
  ).pipe(
    Effect.mapError(() => new InvalidReleaseManifest({ message: "release manifest is malformed" })),
    Effect.flatMap(validateReleaseManifest),
  )

export const validateReleaseManifest = (
  manifest: ReleaseManifest,
): Effect.Effect<ReleaseManifest, InvalidReleaseManifest> => {
  const ids = new Set<string>()
  const names = new Set<string>()
  const fail = (message: string) => Effect.fail(new InvalidReleaseManifest({ message }))
  if (manifest.tag !== releaseTag(manifest.version)) {
    return fail("release tag does not match version")
  }
  const pluginNames = new Set<string>()
  const pluginHosts = new Set<PluginHost>()
  for (const plugin of manifest.plugins) {
    if (plugin.rpcVersion !== manifest.rpc.version || pluginNames.has(plugin.name) || pluginHosts.has(plugin.host)) return fail("plugin selection does not match the release RPC version")
    pluginNames.add(plugin.name)
    pluginHosts.add(plugin.host)
  }
  if (PluginHostSchema.literals.some((host) => !pluginHosts.has(host))) return fail("release is missing a plugin selection for a declared host")
  for (const artifact of manifest.artifacts) {
    if (ids.has(artifact.id) || names.has(artifact.filename)) {
      return fail("release artifact IDs and filenames must be unique")
    }
    ids.add(artifact.id)
    names.add(artifact.filename)
    if (Option.isNone(artifact.host)) {
      return fail(`${artifact.id} has invalid host metadata`)
    }
    const icn = artifact.kind === "icn-base" || artifact.kind === "icn-backend"
    if (icn !== (Option.isSome(artifact.nativeBuild) && Option.isSome(artifact.backendModuleAbi))) {
      return fail(`${artifact.id} has invalid native identity metadata`)
    }
    if (artifact.kind === "icn-base" && (
      Option.getOrUndefined(artifact.backend) !== "cpu" ||
      Option.isSome(artifact.requiredBaseId) ||
      Option.isSome(artifact.compatibility)
    )) {
      return fail(`${artifact.id} is not a CPU base`)
    }
    if (artifact.kind === "icn-backend" && (
      Option.isNone(artifact.backend) ||
      artifact.backend.value === "cpu" ||
      Option.isNone(artifact.requiredBaseId) ||
      Option.isNone(artifact.compatibility)
    )) {
      return fail(`${artifact.id} has incomplete backend-pack metadata`)
    }
  }
  for (const artifact of manifest.artifacts) {
    if (Option.isSome(artifact.requiredBaseId) && !ids.has(artifact.requiredBaseId.value)) {
      return fail(`${artifact.id} references a missing base`)
    }
  }
  return Effect.succeed(manifest)
}
