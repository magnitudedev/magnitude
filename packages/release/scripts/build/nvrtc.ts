import * as Command from "@effect/platform/Command"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect, Option, Schema } from "effect"
import {
  defaultArtifactDownloadPolicy,
  downloadArtifact,
} from "../../src/artifact-download"
import type { NvrtcRedistributable } from "../../src/targets"

export class NvrtcStagingFailed extends Schema.TaggedError<NvrtcStagingFailed>()("NvrtcStagingFailed", {
  message: Schema.String,
}) {}

const stagingFailure = (message: string) => (cause: unknown) =>
  new NvrtcStagingFailed({ message: `${message}: ${String(cause)}` })

/** NVRTC's files for the installation's `runtime/` directory, under their installed names. */
export interface StagedNvrtc {
  readonly libraries: readonly string[]
  readonly license: string
}

/**
 * Downloads NVIDIA's pinned NVRTC redistributable, verifies its size and SHA-256, and stages the
 * two libraries and the license notice under their installed names. Staged files are cached by
 * archive digest; a cache entry is published by rename only once complete.
 */
export const stageNvrtc = (
  nvrtc: NvrtcRedistributable,
  cacheRoot: string,
): Effect.Effect<StagedNvrtc, NvrtcStagingFailed, FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    const files = [...nvrtc.libraries, nvrtc.license]
    const staged = path.join(cacheRoot, nvrtc.sha256)
    const stagedFiles: StagedNvrtc = {
      libraries: nvrtc.libraries.map((file) => path.join(staged, file.name)),
      license: path.join(staged, nvrtc.license.name),
    }
    if (yield* fs.exists(staged).pipe(Effect.mapError(stagingFailure(`unable to inspect ${staged}`)))) {
      const complete = yield* Effect.forEach(
        [...stagedFiles.libraries, stagedFiles.license],
        (file) => fs.stat(file).pipe(
          Effect.option,
          Effect.map((info) => Option.isSome(info) && info.value.type === "File" && Number(info.value.size) > 0),
        ),
      ).pipe(Effect.map((present) => present.every(Boolean)))
      if (complete) return stagedFiles
      yield* fs.remove(staged, { recursive: true }).pipe(
        Effect.mapError(stagingFailure(`unable to remove incomplete NVRTC staging at ${staged}`)),
      )
    }
    yield* fs.makeDirectory(cacheRoot, { recursive: true }).pipe(
      Effect.mapError(stagingFailure(`unable to create ${cacheRoot}`)),
    )
    yield* Effect.acquireUseRelease(
      fs.makeTempDirectory({ directory: cacheRoot, prefix: ".nvrtc-" }).pipe(
        Effect.mapError(stagingFailure("unable to create an NVRTC staging directory")),
      ),
      (scratch) => Effect.gen(function* () {
        const archive = path.join(scratch, path.basename(nvrtc.url))
        yield* downloadArtifact({
          url: nvrtc.url,
          destination: archive,
          bytes: nvrtc.bytes,
          sha256: nvrtc.sha256,
          strategy: { _tag: "Sequential" },
          policy: defaultArtifactDownloadPolicy,
          onProgress: Option.none(),
          onVerificationProgress: Option.none(),
        }).pipe(
          Effect.provide(FetchHttpClient.layer),
          Effect.mapError((error) => new NvrtcStagingFailed({ message: `NVRTC ${nvrtc.version} download failed: ${error.message}` })),
        )
        const extracted = path.join(scratch, "extracted")
        const runtime = path.join(scratch, "runtime")
        yield* fs.makeDirectory(extracted)
        yield* fs.makeDirectory(runtime)
        // bsdtar (macOS, Windows) and GNU tar both read NVIDIA's .tar.xz and bsdtar reads its .zip.
        const exitCode = yield* Command.make("tar", "-xf", archive, "-C", extracted, ...files.map((file) => file.member)).pipe(
          Command.stderr("inherit"),
          Command.exitCode,
        )
        if (exitCode !== 0) {
          return yield* new NvrtcStagingFailed({ message: `unable to extract NVRTC from ${path.basename(archive)} (tar exited ${exitCode})` })
        }
        for (const file of files) {
          yield* fs.copyFile(path.join(extracted, file.member), path.join(runtime, file.name))
        }
        yield* fs.rename(runtime, staged)
      }).pipe(Effect.mapError((cause) =>
        cause instanceof NvrtcStagingFailed ? cause : stagingFailure(`unable to stage NVRTC ${nvrtc.version}`)(cause))),
      (scratch) => fs.remove(scratch, { recursive: true, force: true }).pipe(Effect.orDie),
    )
    return stagedFiles
  })
