/**
 * Writes the Docker build context for one architecture of a release:
 *
 *   magnitude-desktop.deb       the release's Debian package, or the one given with --deb
 *   inference/                  the release's inference engine with its installation.json
 *   magnitude-release.json      the release manifest every artifact was checked against
 *   update-configuration.json   only with --update-configuration (QA images)
 *
 * Usage:
 *   bun packages/release/scripts/docker/prepare-context.ts --version 0.2.14 --arch amd64 \
 *     --output ./docker-context [--candidate ./release-candidate | --deb ./magnitude-desktop.deb] \
 *     [--update-configuration ./update.json]
 *
 * The release is the published one named by --version, or with --candidate the not yet published
 * candidate that assemble.ts wrote. With --deb, only the engine comes from the release.
 */
import { FetchHttpClient, FileSystem, Path } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { IcnInstallationDeclaration } from "@magnitudedev/icn-protocol"
import { Data, Effect, Layer, Option, Schema } from "effect"
import { parseArgs } from "node:util"
import { acquireRelease, installArtifact, releaseBaseUrl, releaseUrl, selectArtifact, validateReleaseManifestBytes } from "../../src/acquisition"
import { ArchiveExtractor, NodeArchiveExtractor } from "../../src/archive"
import { defaultArtifactDownloadPolicy, downloadArtifact } from "../../src/artifact-download"
import { inferenceRequiredPaths } from "../../src/inference-installation"
import { sha256File } from "../../src/macos-app"
import { hostById } from "../../src/targets"
import type { ReleaseArtifact } from "../../src/contracts"
import { debArtifactId, parseDockerArchitecture, releaseHostOf } from "./image"

class DockerContextFailed extends Data.TaggedError("DockerContextFailed")<{ readonly message: string }> {}

const fail = (message: string) => new DockerContextFailed({ message })

const readArguments = Effect.try({
  try: () => parseArgs({
    args: process.argv.slice(2),
    options: {
      version: { type: "string" },
      arch: { type: "string" },
      output: { type: "string" },
      candidate: { type: "string" },
      deb: { type: "string" },
      "update-configuration": { type: "string" },
    },
    strict: true,
  }).values,
  catch: (cause) => fail(cause instanceof Error ? cause.message : String(cause)),
})

const required = (value: string | undefined, name: string) =>
  value === undefined || value.trim() === "" ? Effect.fail(fail(`--${name} is required`)) : Effect.succeed(value.trim())

const run = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const args = yield* readArguments
  const version = yield* required(args.version, "version")
  const architecture = yield* parseDockerArchitecture(yield* required(args.arch, "arch"))
  const output = path.resolve(yield* required(args.output, "output"))
  const candidate = Option.map(Option.fromNullable(args.candidate), (directory) => path.resolve(directory))
  const localDeb = Option.map(Option.fromNullable(args.deb), (file) => path.resolve(file))
  if (Option.isSome(candidate) && Option.isSome(localDeb)) return yield* fail("--candidate already holds the Debian package; omit --deb")
  const updateConfiguration = Option.map(Option.fromNullable(args["update-configuration"]), (file) => path.resolve(file))
  const host = releaseHostOf(architecture)
  const baseUrl = releaseBaseUrl()

  if ((yield* fs.exists(output)) && (yield* fs.readDirectory(output)).length > 0) {
    return yield* fail(`${output} must be empty`)
  }
  yield* fs.makeDirectory(output, { recursive: true })

  const cache = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-docker-release-" })
  const manifestDirectory = Option.getOrElse(candidate, () => path.join(cache, "manifest"))
  const release = yield* Option.match(candidate, {
    onNone: () => acquireRelease(baseUrl, version, manifestDirectory),
    onSome: (directory) => fs.readFile(path.join(directory, "magnitude-release.json")).pipe(Effect.flatMap(validateReleaseManifestBytes)),
  })
  if (release.manifest.version !== version) return yield* fail(`The release manifest is for ${release.manifest.version}, not ${version}`)
  yield* fs.copyFile(path.join(manifestDirectory, "magnitude-release.json"), path.join(output, "magnitude-release.json"))
  // A candidate's artifacts are files beside its manifest, checked against it before use.
  const candidateFile = (directory: string, artifact: ReleaseArtifact) => Effect.gen(function* () {
    const file = path.join(directory, artifact.filename)
    const size = yield* fs.stat(file).pipe(Effect.map((info) => Number(info.size)))
    if (size !== artifact.bytes || (yield* sha256File(file)) !== artifact.sha256) return yield* fail(`${artifact.filename} differs from the candidate manifest`)
    return file
  })

  // The engine goes first: its installation clears leftover staging entries beside it.
  const engine = yield* selectArtifact(release.manifest, "icn-base", host)
  const nativeBuild = yield* Option.match(engine.nativeBuild, {
    onNone: () => Effect.fail(fail(`${engine.id} names no native build`)),
    onSome: Effect.succeed,
  })
  yield* Effect.logInfo(`Installing ${engine.filename} (${engine.bytes} bytes)`)
  const inference = path.join(output, "inference")
  yield* Option.match(candidate, {
    onNone: () => installArtifact(baseUrl, version, engine, inference),
    onSome: (directory) => candidateFile(directory, engine).pipe(Effect.flatMap((archive) =>
      fs.makeDirectory(inference).pipe(Effect.zipRight(ArchiveExtractor.pipe(Effect.flatMap((extractor) =>
        extractor.extract(archive, inference, engine, Option.none()))))))),
  })
  for (const file of inferenceRequiredPaths(hostById(host))) {
    if (!(yield* fs.exists(path.join(inference, file)))) return yield* fail(`${engine.filename} has no ${file}`)
  }
  const declaration = yield* Schema.encode(Schema.parseJson(IcnInstallationDeclaration))({ schemaVersion: 1, nativeBuild })
  yield* fs.writeFileString(path.join(inference, "installation.json"), `${declaration}\n`)

  const deb = path.join(output, "magnitude-desktop.deb")
  const debArtifact = Effect.suspend(() => {
    const artifact = release.manifest.artifacts.find((entry) => entry.id === debArtifactId(architecture))
    return artifact === undefined ? Effect.fail(fail(`${version} has no ${debArtifactId(architecture)} artifact`)) : Effect.succeed(artifact)
  })
  if (Option.isSome(localDeb)) {
    yield* Effect.logInfo(`Using the local package ${localDeb.value}`)
    yield* fs.copyFile(localDeb.value, deb)
  } else if (Option.isSome(candidate)) {
    yield* fs.copyFile(yield* candidateFile(candidate.value, yield* debArtifact), deb)
  } else {
    const artifact = yield* debArtifact
    yield* Effect.logInfo(`Downloading ${artifact.filename} (${artifact.bytes} bytes)`)
    yield* downloadArtifact({
      url: releaseUrl(baseUrl, version, artifact.filename),
      destination: deb,
      bytes: artifact.bytes,
      sha256: artifact.sha256,
      strategy: { _tag: "Sequential" },
      policy: defaultArtifactDownloadPolicy,
      onProgress: Option.none(),
      onVerificationProgress: Option.none(),
    })
  }

  if (Option.isSome(updateConfiguration)) {
    const text = yield* fs.readFileString(updateConfiguration.value)
    yield* Schema.decodeUnknown(Schema.parseJson(Schema.Record({ key: Schema.String, value: Schema.Unknown })))(text)
    yield* fs.writeFileString(path.join(output, "update-configuration.json"), text)
  }

  yield* Effect.logInfo("Docker build context prepared", {
    output, version, architecture, engine: engine.filename, nativeBuild, sourceCommit: release.manifest.sourceCommit,
  })
}))

BunRuntime.runMain(run.pipe(Effect.provide(Layer.mergeAll(NodeArchiveExtractor, FetchHttpClient.layer, BunContext.layer))))
