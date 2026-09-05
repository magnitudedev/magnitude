import * as CommandExecutor from "@effect/platform/CommandExecutor"

import * as FileSystem from "@effect/platform/FileSystem"

import * as HttpClient from "@effect/platform/HttpClient"

import * as Path from "@effect/platform/Path"

import type { AcnTarget } from "@magnitudedev/acn-protocol"

import type { ArtifactInstallationEvent } from "@magnitudedev/release"

import { Array as Arr, Context, Effect, Option, Ref } from "effect"

import { resolveBinaryCommand, type BinaryAcquisitionEvent } from "../binary"

import { DAEMON_TARGET } from "../version"

import type { AcnEnsureEvent } from "./acn-instance-manager"

import {
  AcnDaemonTargetUnsupported,
  AcnLaunchOverrideTargetMismatch,
  type AcnEnsuranceError,
} from "./errors"


export interface AcnLaunchOverride {
  readonly target: AcnTarget
  readonly command: Arr.NonEmptyReadonlyArray<string>
}


export interface AcnDaemonLaunchCommand {
  readonly target: AcnTarget
  readonly command: Arr.NonEmptyReadonlyArray<string>
}


export interface AcnDaemonLaunchCommandResolver {
  readonly resolve: (
    target: AcnTarget,
    report: (event: AcnEnsureEvent) => Effect.Effect<void>,
  ) => Effect.Effect<AcnDaemonLaunchCommand, AcnEnsuranceError>
}


export const AcnDaemonLaunchCommandResolver = Context.GenericTag<AcnDaemonLaunchCommandResolver>(
  "@magnitudedev/daemon-management/AcnDaemonLaunchCommandResolver",
)


const sameTarget = (left: AcnTarget, right: AcnTarget): boolean =>
  left.revision === right.revision && left.identity === right.identity


const artifactProgress = (
  event: Extract<ArtifactInstallationEvent, { readonly _tag: "Downloading" }>,
) => ({
  completed: event.progress.acceptedBytes,
  totalBytes: event.progress.totalBytes,
  unit: "Bytes" as const,
  attempt: Option.some(event.progress.attempt),
})


export interface MakeAcnDaemonLaunchCommandResolverOptions {
  readonly binaryPath: string | undefined
  readonly dataDirectory: string
  readonly launchOverride: AcnLaunchOverride | undefined
  readonly fileSystem: FileSystem.FileSystem
  readonly httpClient: HttpClient.HttpClient
  readonly commandExecutor: CommandExecutor.CommandExecutor
  readonly path: Path.Path
}


export const makeAcnDaemonLaunchCommandResolver = (
  options: MakeAcnDaemonLaunchCommandResolverOptions,
): AcnDaemonLaunchCommandResolver => {
  const resolve: AcnDaemonLaunchCommandResolver["resolve"] = (target, reportEvent) => {
    if (!sameTarget(target, DAEMON_TARGET)) {
      return Effect.fail(new AcnDaemonTargetUnsupported({
        requested: target,
        supported: DAEMON_TARGET,
      }))
    }
    if (options.launchOverride !== undefined) {
      return sameTarget(options.launchOverride.target, target)
        ? Effect.succeed(options.launchOverride)
        : Effect.fail(new AcnLaunchOverrideTargetMismatch({
            requested: target,
            override: options.launchOverride.target,
          }))
    }
    return Effect.gen(function* () {
      const plan = yield* Ref.make(Option.none<{
        readonly daemonBytes: number
        readonly inferenceEngineBytes: number
        readonly inferenceEngineBytesExact: boolean
      }>())
      const report = (event: BinaryAcquisitionEvent) => Effect.gen(function* () {
        if (event._tag === "Planned") {
          yield* Ref.set(plan, Option.some(event.plan))
          return
        }
        if (event.event._tag !== "Downloading") return
        const download = event.event
        const currentPlan = yield* Ref.get(plan)
        if (Option.isNone(currentPlan)) return
        yield* reportEvent({
          _tag: "Observation",
          observation: {
            _tag: "Installing",
            phase: "DownloadingDaemon",
            plan: currentPlan.value,
            progress: Option.some(artifactProgress(download)),
          },
        })
      })
      return yield* resolveBinaryCommand({
        binaryPath: options.binaryPath,
        version: target.identity,
        acnRevision: target.revision,
        dataDir: options.dataDirectory,
        acquisitionObserver: Option.some({ report }),
      }).pipe(
        Effect.provideService(FileSystem.FileSystem, options.fileSystem),
        Effect.provideService(HttpClient.HttpClient, options.httpClient),
        Effect.provideService(CommandExecutor.CommandExecutor, options.commandExecutor),
        Effect.provideService(Path.Path, options.path),
        Effect.map((resolved) => ({ target, command: resolved.command })),
      )
    })
  }
  return AcnDaemonLaunchCommandResolver.of({ resolve })
}
