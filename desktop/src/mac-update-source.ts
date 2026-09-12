import { FetchHttpClient, FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { defaultArtifactDownloadPolicy, downloadArtifact, findMacDesktopUpdate, RELEASE_DIST_TAGS_URL, releaseUrl } from "@magnitudedev/release"
import { Effect, Option } from "effect"
import { join } from "node:path"
import { ApplicationUpdateFailed, ApplicationUpdateSource } from "./application-update"
import { NativeMacUpdate, stageMacUpdateArchive } from "./mac-update-stage"
import { ApplicationUpdateHandoff } from "./update-handoff"

export const macUpdateSource = (options: {
  readonly currentVersion: string
  readonly host: "darwin-arm64" | "darwin-x64"
  readonly releaseBaseUrl: string
  readonly cacheDirectory: string
}) => Effect.gen(function* () {
  const native = yield* NativeMacUpdate
  const handoff = yield* ApplicationUpdateHandoff
  const failure = (error: { readonly message?: string; readonly reason?: string }) => new ApplicationUpdateFailed({
    message: error.message ?? error.reason ?? "Could not download the application update.",
  })
  return ApplicationUpdateSource.of({
    check: findMacDesktopUpdate({ ...options, registryUrl: RELEASE_DIST_TAGS_URL }).pipe(
      Effect.mapError(failure), Effect.provide([NodeContext.layer, FetchHttpClient.layer]),
    ),
    download: (candidate, progress) => Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      yield* fs.makeDirectory(options.cacheDirectory, { recursive: true, mode: 0o700 })
      const directory = yield* fs.makeTempDirectoryScoped({ directory: options.cacheDirectory, prefix: "desktop-update-" })
      const downloaded = yield* downloadArtifact({
        url: releaseUrl(options.releaseBaseUrl, candidate.version, candidate.artifact.filename),
        destination: join(directory, candidate.artifact.filename),
        bytes: candidate.artifact.bytes, sha256: candidate.artifact.sha256,
        strategy: { _tag: "Sequential" }, policy: defaultArtifactDownloadPolicy,
        onProgress: Option.some(value => progress(value.acceptedBytes)), onVerificationProgress: Option.none(),
      })
      return downloaded.destination
    }).pipe(Effect.mapError(failure), Effect.provide([NodeContext.layer, FetchHttpClient.layer])),
    stage: (archive, candidate) => handoff.record(candidate.version).pipe(
      Effect.zipRight(stageMacUpdateArchive(archive).pipe(Effect.provideService(NativeMacUpdate, native))), Effect.mapError(failure),
    ),
  })
})
