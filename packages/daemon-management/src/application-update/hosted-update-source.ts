import { FetchHttpClient, FileSystem, Path } from "@effect/platform"
import { checkHostedUpdate, resolveHostedDownload, downloadUpdateArtifact, updateInstallerFilename, ReleaseTarget, type HostedUpdateConnection } from "@magnitudedev/release/hosted-update"
import { Effect, Option, Schema } from "effect"
import type { KeyObject } from "node:crypto"
import { join } from "node:path"
import { ApplicationUpdateFailed, ApplicationUpdateSource } from "./application-update"

export type HostedUpdateSourceOptions = HostedUpdateConnection & {
  readonly trustedPublishers: ReadonlyMap<string, KeyObject>
  readonly dataDirectory: string
}

/** All native installers consume the same authenticated, checksum-verified transfer. */
export const hostedUpdateSource = (options: HostedUpdateSourceOptions, stage: ApplicationUpdateSource["stage"]) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  return ApplicationUpdateSource.of({
    check: checkHostedUpdate(options).pipe(
      Effect.mapError(() => new ApplicationUpdateFailed({ message: "Could not check for application updates." })), Effect.provide(FetchHttpClient.layer),
    ),
    download: (candidate, progress) => Effect.gen(function* () {
      const url = yield* resolveHostedDownload({ ...options, release: candidate })
      const transfers = join(options.dataDirectory, "update-downloads")
      yield* fs.makeDirectory(transfers, { recursive: true, mode: 0o700 })
      const directory = yield* fs.makeTempDirectoryScoped({ directory: transfers, prefix: "desktop-update-" })
      const downloaded = yield* downloadUpdateArtifact({
        url, destination: join(directory, updateInstallerFilename(yield* Schema.decodeUnknown(ReleaseTarget)({ os: options.metadata.os, arch: options.metadata.arch, package: options.metadata.package }))),
        release: candidate,
        onProgress: Option.some(value => progress(value.acceptedBytes)),
      })
      return downloaded.destination
  }).pipe(Effect.mapError(() => new ApplicationUpdateFailed({ message: "Could not download and verify the application update." })), Effect.provideService(FileSystem.FileSystem, fs), Effect.provideService(Path.Path, path), Effect.provide(FetchHttpClient.layer)),
  stage,
  })
})
