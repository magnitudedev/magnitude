import { Data, Effect, Option, Schema } from "effect"
import { PluginArtifactSchema, RpcReleaseSchema, type PluginHost } from "./plugins"

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
const releaseArtifactFields = {
  id: NonEmpty,
  kind: Schema.Literal("cli", "acn", "desktop", "icn-base"),
  host: Schema.optionalWith(Host, { as: "Option", exact: true }),
  filename: NonEmpty,
  bytes: PositiveInt,
  sha256: Sha256,
  nativeBuild: Schema.optionalWith(NonEmpty, { as: "Option", exact: true }),
}

export const ReleaseArtifactSchema = Schema.Struct(releaseArtifactFields)
export type ReleaseArtifact = typeof ReleaseArtifactSchema.Type

/**
 * Shipped desktops decode the next release's manifest before they can update. Their decoder
 * requires schema 2 and a CPU `backend` plus a `backendModuleAbi` on `icn-base`, so the manifest
 * keeps those fixed values on the wire. Nothing in this release reads them.
 */
const INFERENCE_ARTIFACT_BACKEND = "cpu"
const INFERENCE_ARTIFACT_MODULE_ABI = "single-artifact"

const ReleaseArtifactWireSchema = Schema.transform(
  Schema.Struct({
    ...releaseArtifactFields,
    backend: Schema.optionalWith(Schema.Literal(INFERENCE_ARTIFACT_BACKEND), { as: "Option", exact: true }),
    backendModuleAbi: Schema.optionalWith(Schema.Literal(INFERENCE_ARTIFACT_MODULE_ABI), { as: "Option", exact: true }),
  }),
  Schema.typeSchema(ReleaseArtifactSchema),
  {
    strict: true,
    decode: ({ backend: _backend, backendModuleAbi: _backendModuleAbi, ...artifact }) => artifact,
    encode: (artifact) => {
      const inference = artifact.kind === "icn-base"
      return {
        ...artifact,
        backend: inference
          ? Option.some<typeof INFERENCE_ARTIFACT_BACKEND>(INFERENCE_ARTIFACT_BACKEND)
          : Option.none<typeof INFERENCE_ARTIFACT_BACKEND>(),
        backendModuleAbi: inference
          ? Option.some<typeof INFERENCE_ARTIFACT_MODULE_ABI>(INFERENCE_ARTIFACT_MODULE_ABI)
          : Option.none<typeof INFERENCE_ARTIFACT_MODULE_ABI>(),
      }
    },
  },
)

export const ReleaseManifestSchema = Schema.Struct({
  schemaVersion: Schema.Literal(2),
  version: NonEmpty,
  acnRevision: PositiveInt.pipe(Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER)),
  tag: NonEmpty,
  sourceCommit: Schema.String.pipe(Schema.pattern(/^[a-f0-9]{40}$/)),
  rpc: RpcReleaseSchema,
  plugins: Schema.Array(PluginArtifactSchema),
  artifacts: Schema.NonEmptyArray(ReleaseArtifactWireSchema),
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
  for (const artifact of manifest.artifacts) {
    if (ids.has(artifact.id) || names.has(artifact.filename)) {
      return fail("release artifact IDs and filenames must be unique")
    }
    ids.add(artifact.id)
    names.add(artifact.filename)
    if (Option.isNone(artifact.host)) {
      return fail(`${artifact.id} has invalid host metadata`)
    }
    if ((artifact.kind === "icn-base") !== Option.isSome(artifact.nativeBuild)) {
      return fail(`${artifact.id} has invalid native identity metadata`)
    }
  }
  return Effect.succeed(manifest)
}
