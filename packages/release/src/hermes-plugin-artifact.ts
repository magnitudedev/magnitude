import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Option, Schema, Stream } from "effect"
import { resolve } from "node:path"
import { prerelease, rcompare, valid } from "semver"
import { canonical } from "@magnitudedev/utils/canonical-key"
import { PluginArtifactError } from "./plugin-artifacts"
import { stageHermesPluginContent, verifyHermesPluginContent } from "./hermes-plugin-content"
import { HermesPluginArtifactSchema, type HermesPluginArtifact } from "./plugins"

export const HERMES_PLUGIN_REPOSITORY = "https://github.com/magnitudedev/magnitude"
export const hermesPluginTag = (version: string) => `refs/tags/magnitude-hermes/${version}`

const git = (directory: string, args: readonly string[], inheritConfig = false) => Effect.scoped(Effect.gen(function* () {
  const child = yield* Command.make("git", ...args).pipe(
  Command.workingDirectory(directory),
  Command.env({
    ...(inheritConfig ? {} : { GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null" }),
    GIT_AUTHOR_NAME: "Magnitude Release", GIT_AUTHOR_EMAIL: "release@magnitude.dev",
    GIT_COMMITTER_NAME: "Magnitude Release", GIT_COMMITTER_EMAIL: "release@magnitude.dev",
    GIT_AUTHOR_DATE: "2000-01-01T00:00:00Z", GIT_COMMITTER_DATE: "2000-01-01T00:00:00Z",
    GIT_TERMINAL_PROMPT: "0",
  }),
    Command.start,
  )
  const text = <E, R>(stream: Stream.Stream<Uint8Array, E, R>) => stream.pipe(
    Stream.decodeText(), Stream.runFold("", (text, chunk) => (text + chunk).slice(-65_536)),
  )
  const [code, stdout, stderr] = yield* Effect.all([child.exitCode, text(child.stdout), text(child.stderr)], { concurrency: "unbounded" })
  if (Number(code) !== 0) return yield* new PluginArtifactError({ message: `Git ${args[0]} failed (${code}): ${stderr}` })
  return stdout.trim()
})).pipe(Effect.timeout("2 minutes"))

/** A deterministic, parentless distribution commit containing only consumer files. */
export const packHermesPlugin = (directory: string, output: string) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hermes-artifact-" })
  const checkout = resolve(temporary, "package")
  const { metadata } = yield* stageHermesPluginContent(directory, checkout)
  yield* git(checkout, ["init", "--initial-branch=main", "--template="])
  yield* git(checkout, ["-c", "core.autocrlf=false", "add", "--all"])
  yield* git(checkout, ["-c", "commit.gpgSign=false", "commit", "-m", `Magnitude for Hermes ${metadata.version}`])
  const revision = yield* git(checkout, ["rev-parse", "HEAD"])
  const filename = `magnitudedev-hermes-companion-${metadata.version}.bundle`
  yield* fs.makeDirectory(output, { recursive: true })
  const bundle = resolve(output, filename)
  yield* git(checkout, ["bundle", "create", bundle, "refs/heads/main"])
  return yield* Schema.decodeUnknown(HermesPluginArtifactSchema)({
    host: "hermes", name: metadata.name, version: metadata.version, rpcVersion: metadata.rpcVersion,
    contentFingerprint: metadata.contentFingerprint, filename, revision,
    repository: HERMES_PLUGIN_REPOSITORY,
  })
}))

/** Acceptance and publication both verify the exact Git payload, not a rebuilt tree. */
export const verifyHermesPluginArtifact = (artifact: HermesPluginArtifact, directory: string) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const bundle = resolve(directory, artifact.filename)
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hermes-verification-" })
  const refs = yield* git(temporary, ["bundle", "list-heads", bundle])
  if (refs !== `${artifact.revision} refs/heads/main`) return yield* new PluginArtifactError({
    message: "Hermes Git bundle must expose only the selected distribution commit",
  })
  yield* git(temporary, ["clone", "--no-checkout", bundle, "package"])
  const checkout = resolve(temporary, "package")
  if ((yield* git(checkout, ["rev-parse", "refs/remotes/origin/main"])) !== artifact.revision) {
    return yield* new PluginArtifactError({ message: "Hermes Git bundle revision differs from the selected commit" })
  }
  yield* git(checkout, ["checkout", "--detach", artifact.revision])
  const { metadata } = yield* verifyHermesPluginContent(checkout)
  if (metadata.name !== artifact.name || metadata.version !== artifact.version
    || metadata.rpcVersion !== artifact.rpcVersion || metadata.contentFingerprint !== artifact.contentFingerprint) {
    return yield* new PluginArtifactError({ message: "Hermes Git bundle content differs from the prepared release" })
  }
  // Reconstruct the commit from the verified consumer tree to reject extra
  // files. Git's pack representation may vary; the content-addressed commit
  // is release identity, while the acceptance receipt binds actual bundle bytes.
  const reconstructed = yield* packHermesPlugin(checkout, resolve(temporary, "repacked"))
  if (canonical(reconstructed) !== canonical(artifact)) return yield* new PluginArtifactError({
    message: "Hermes Git bundle contains unaccounted distribution content",
  })
  return bundle
}))

const RemoteRef = Schema.Struct({ revision: HermesPluginArtifactSchema.fields.revision, ref: Schema.NonEmptyString })
const remoteRefs = (repository: string, pattern: string, cwd: string) => git(cwd, ["ls-remote", "--refs", repository, pattern]).pipe(
  Effect.flatMap(output => Schema.decodeUnknown(Schema.Array(RemoteRef))(output === "" ? [] : output.split("\n").map(line => {
    const [revision, ref] = line.split("\t")
    return { revision, ref }
  }))),
)

export const publishedHermesRevision = (artifact: Pick<HermesPluginArtifact, "repository" | "version">, cwd: string) =>
  remoteRefs(artifact.repository, hermesPluginTag(artifact.version), cwd).pipe(Effect.map(refs => Option.fromNullable(refs[0]?.revision)))

/** Hermes has its own public baseline: immutable distribution tags in the package repository. */
export const publishedHermesPlugin = (cwd: string, channel: string) => Effect.scoped(Effect.gen(function* () {
  const refs = yield* remoteRefs(HERMES_PLUGIN_REPOSITORY, hermesPluginTag("*"), cwd)
  const versions = refs.map(ref => ({ ...ref, version: ref.ref.slice(hermesPluginTag("").length) }))
    .filter(({ version }) => valid(version) !== null)
    .sort((left, right) => rcompare(left.version, right.version))
  // Match npm's channel baseline: a new prerelease channel starts from stable,
  // not from an unrelated prerelease or a fictitious first publication.
  const latest = versions.find(({ version }) => channel === "latest"
    ? prerelease(version) === null : String(prerelease(version)?.[0]) === channel)
    ?? versions.find(({ version }) => prerelease(version) === null)
  if (latest === undefined) return Option.none<HermesPluginArtifact>()
  const fs = yield* FileSystem.FileSystem
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hermes-baseline-" })
  yield* git(temporary, ["init", "--initial-branch=main", "--template="])
  yield* git(temporary, ["fetch", HERMES_PLUGIN_REPOSITORY, latest.ref])
  yield* git(temporary, ["checkout", "--detach", "FETCH_HEAD"])
  const artifact = yield* packHermesPlugin(temporary, resolve(temporary, "artifact"))
  if (artifact.revision !== latest.revision || artifact.version !== latest.version) return yield* new PluginArtifactError({
    message: "Published Hermes tag does not contain the expected distribution package",
  })
  return Option.some(artifact)
}))

/** Publish the accepted commit under an immutable tag; never force or rebuild. */
export const publishHermesPlugin = (artifact: HermesPluginArtifact, directory: string, cwd: string) => Effect.gen(function* () {
  const bundle = yield* verifyHermesPluginArtifact(artifact, directory)
  const existing = yield* publishedHermesRevision(artifact, cwd)
  if (Option.isSome(existing) && existing.value !== artifact.revision) return yield* new PluginArtifactError({
    message: `Hermes ${artifact.version} is already published at a different commit; never overwrite a version`,
  })
  if (Option.isNone(existing)) {
    // The release checkout supplies its normal GitHub authentication. Fetching
    // the accepted bundle imports objects without changing its branch or files.
    yield* git(cwd, ["fetch", "--no-tags", bundle, artifact.revision], true)
    yield* git(cwd, ["push", artifact.repository, `${artifact.revision}:${hermesPluginTag(artifact.version)}`], true)
  }
  if (!Option.contains(artifact.revision)(yield* publishedHermesRevision(artifact, cwd))) {
    return yield* new PluginArtifactError({ message: "The public Hermes tag does not resolve to the accepted commit" })
  }
})

/** Retrieve an unchanged Git payload without rebuilding any Python or Desktop code. */
export const acquireHermesPluginArtifact = (artifact: HermesPluginArtifact, output: string) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hermes-acquisition-" })
  yield* git(temporary, ["init", "--initial-branch=main", "--template="])
  yield* git(temporary, ["fetch", artifact.repository, artifact.revision])
  yield* git(temporary, ["checkout", "--detach", "FETCH_HEAD"])
  const actual = yield* packHermesPlugin(temporary, output)
  if (canonical(actual) !== canonical(artifact)) return yield* new PluginArtifactError({
    message: "Published Hermes package differs from the selected immutable artifact",
  })
  return yield* verifyHermesPluginArtifact(artifact, output)
}))
