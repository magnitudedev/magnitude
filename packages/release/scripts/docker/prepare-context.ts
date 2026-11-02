/**
 * Writes the Docker build context for one architecture of a published release:
 *
 *   magnitude-desktop.deb       the release's Debian package, or the one given with --deb
 *   inference/                  the release's inference engine with its installation.json
 *   magnitude-release.json      the release manifest every download was checked against
 *   update-configuration.json   only with --update-configuration (QA images)
 *
 * Usage:
 *   bun packages/release/scripts/docker/prepare-context.ts --version 0.2.14 --arch amd64 \
 *     --output ./docker-context [--deb ./magnitude-desktop.deb] [--update-configuration ./update.json]
 *
 * With --deb, only the engine comes from the release named by --version.
 */
import { FetchHttpClient, FileSystem, Path } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { IcnInstallationDeclaration } from "@magnitudedev/icn-protocol"
import { Data, Effect, Layer, Option, Schema } from "effect"
import { parseArgs } from "node:util"
import { acquireRelease, installArtifact, releaseBaseUrl, releaseUrl, selectArtifact } from "../../src/acquisition"
import { NodeArchiveExtractor } from "../../src/archive"
import { defaultArtifactDownloadPolicy, downloadArtifact } from "../../src/artifact-download"
import { inferenceRequiredPaths } from "../../src/inference-installation"
import { hostById } from "../../src/targets"
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
  const localDeb = Option.map(Option.fromNullable(args.deb), (file) => path.resolve(file))
  const updateConfiguration = Option.map(Option.fromNullable(args["update-configuration"]), (file) => path.resolve(file))
  const host = releaseHostOf(architecture)
  const baseUrl = releaseBaseUrl()

  if ((yield* fs.exists(output)) && (yield* fs.readDirectory(output)).length > 0) {
    return yield* fail(`${output} must be empty`)
  }
  yield* fs.makeDirectory(output, { recursive: true })

  const cache = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-docker-release-" })
  const release = yield* acquireRelease(baseUrl, version, path.join(cache, "manifest"))
  yield* fs.copyFile(path.join(cache, "manifest", "magnitude-release.json"), path.join(output, "magnitude-release.json"))

  // The engine goes first: its installation clears leftover staging entries beside it.
  const engine = yield* selectArtifact(release.manifest, "icn-base", host)
  const nativeBuild = yield* Option.match(engine.nativeBuild, {
    onNone: () => Effect.fail(fail(`${engine.id} names no native build`)),
    onSome: Effect.succeed,
  })
  yield* Effect.logInfo(`Installing ${engine.filename} (${engine.bytes} bytes)`)
  const inference = yield* installArtifact(baseUrl, version, engine, path.join(output, "inference"))
  for (const file of inferenceRequiredPaths(hostById(host))) {
    if (!(yield* fs.exists(path.join(inference, file)))) return yield* fail(`${engine.filename} has no ${file}`)
  }
  const declaration = yield* Schema.encode(Schema.parseJson(IcnInstallationDeclaration))({ schemaVersion: 1, nativeBuild })
  yield* fs.writeFileString(path.join(inference, "installation.json"), `${declaration}\n`)

  const deb = path.join(output, "magnitude-desktop.deb")
  if (Option.isSome(localDeb)) {
    yield* Effect.logInfo(`Using the local package ${localDeb.value}`)
    yield* fs.copyFile(localDeb.value, deb)
  } else {
    const artifact = release.manifest.artifacts.find((candidate) => candidate.id === debArtifactId(architecture))
    if (artifact === undefined) return yield* fail(`${version} has no ${debArtifactId(architecture)} artifact`)
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
