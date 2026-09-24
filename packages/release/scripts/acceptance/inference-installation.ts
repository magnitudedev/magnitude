import { FetchHttpClient, FileSystem } from "@effect/platform"
import { IcnInstallationDeclaration } from "@magnitudedev/icn-protocol"
import { Config, Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { resolveReleaseIcnInstallation } from "../../../icn/src/lifecycle/release-installation"
import { IcnPreparationReporter } from "../../../icn/src/lifecycle/preparation"
import { acquireRelease, installArtifact, releaseBaseUrl, selectArtifact } from "../../src/acquisition"
import { NodeArchiveExtractor } from "../../src/archive"
import { currentHost } from "../../src/targets"

/** Update fixture versions are unpublished; service readiness uses an explicit real engine. */
export const acceptanceInferenceInstallation = (dataDirectory: string) => Effect.gen(function* () {
  const supplied = yield* Config.option(Config.string("MAGNITUDE_ICN_PATH"))
  if (Option.isSome(supplied)) return supplied.value
  const version = yield* Config.string("MAGNITUDE_ACCEPTANCE_ICN_VERSION")
  if (yield* Config.boolean("MAGNITUDE_ACCEPTANCE_CPU_ONLY").pipe(Config.withDefault(false))) {
    // Virtual signing runners have no usable accelerator; execute the real published CPU base.
    return yield* Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const release = yield* acquireRelease(releaseBaseUrl(), version, join(dataDirectory, "manifest"))
      const artifact = yield* selectArtifact(release.manifest, "icn-base", currentHost())
      const root = join(dataDirectory, "cpu")
      yield* installArtifact(releaseBaseUrl(), version, artifact, root)
      const declaration = yield* Schema.decodeUnknown(IcnInstallationDeclaration)({
        schemaVersion: 1, backend: "cpu", nativeBuild: Option.getOrUndefined(artifact.nativeBuild),
        backendModuleAbi: Option.getOrUndefined(artifact.backendModuleAbi),
      })
      const path = join(root, "installation.json")
      yield* fs.writeFileString(path, yield* Schema.encode(Schema.parseJson(IcnInstallationDeclaration))(declaration))
      return path
    }).pipe(Effect.provide([FetchHttpClient.layer, NodeArchiveExtractor]))
  }
  const installation = yield* resolveReleaseIcnInstallation(version, dataDirectory, releaseBaseUrl()).pipe(
    Effect.provideService(IcnPreparationReporter, { report: () => Effect.void }), Effect.provide(FetchHttpClient.layer))
  return installation.declarationPath
})
