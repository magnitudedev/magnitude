import { homedir } from "node:os";
import { join, resolve } from "node:path";
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import * as FileSystem from "@effect/platform/FileSystem"
import { Duration, Effect, Layer, Option, Ref } from "effect";
import {
  type AcnInstallationPlan,
  type AcnStartupProgress,
} from "@magnitudedev/acn-protocol";
import {
  applyAcnProcessCommand,
  readAcnProcessState,
  readProcessStartIdentity,
  type ExactIcnProcess,
} from "@magnitudedev/acn-protocol/process-state"
import type { ArtifactInstallationEvent } from "@magnitudedev/release";
import {
  IcnBinaryResolutionConfig,
  IcnLifecycleConfig,
  IcnProcess,
  makeIcnCatalog,
  makeIcnClient,
  makeIcnDownloads,
  makeIcnProcess,
  makeIcnHardware,
  makeIcnInstalledModels,
  IcnInstancesLive,
  IcnStorageConfig,
  IcnPreparationReporter,
  IcnProcessOwnership,
  IcnLifecycleError,
} from "@magnitudedev/icn";
import { ACN_VERSION } from "../version";
import { resolveHuggingFaceCacheRoots } from "./hugging-face-cache";
import { AcnServiceLifecycle } from "../service-lifecycle";
import { ACN_INSTANCE_ID } from "../identity"

const artifactProgress = (
  artifact: "Base" | "Accelerator",
  event: Extract<ArtifactInstallationEvent, { readonly _tag: "Downloading" }>,
  plan: AcnInstallationPlan
): AcnStartupProgress => {
  const completedBeforeArtifact =
    artifact === "Accelerator"
      ? plan.inferenceEngineBytes - event.progress.totalBytes
      : 0;
  return {
    completed: Math.min(
      plan.inferenceEngineBytes,
      completedBeforeArtifact + event.progress.acceptedBytes
    ),
    totalBytes: plan.inferenceEngineBytes,
    unit: "Bytes",
    attempt: Option.some(event.progress.attempt),
  };
};
const defaultDataDir = () => join(homedir(), ".magnitude");

const binarySource = (dataDir: string) => {
  const explicit = process.env.MAGNITUDE_ICN_PATH?.trim();
  if (explicit) {
    return {
      _tag: "Installation" as const,
      path: explicit,
    };
  }
  if (ACN_VERSION.includes("+dev.")) {
    return {
      _tag: "Installation" as const,
      path: resolve(
        import.meta.dir,
        "../../../../inference/target/development/installation.json"
      ),
    };
  }
  return {
    _tag: "Release" as const,
    version: ACN_VERSION,
    dataDir,
    releaseBaseUrl: (
      process.env.MAGNITUDE_RELEASE_BASE_URL ??
      "https://github.com/magnitudedev/magnitude/releases/download"
    ).replace(/\/+$/, ""),
  };
};

const makeProcess = (dataDir: string) =>
  makeIcnProcess(
    new IcnLifecycleConfig({
      binary: new IcnBinaryResolutionConfig({
        source: binarySource(dataDir),
        supportedApiVersion: 1,
        expectedNativeBuild: Option.none(),
        expectedTarget: Option.none(),
        requiredCapabilities: [
          "hardware",
          "model_catalog",
          "model_installed",
          "model_assessment",
          "model_downloads",
          "model_residency",
          "chat_streaming",
        ],
        probeTimeout: Duration.seconds(10),
      }),
      storage: new IcnStorageConfig({
        modelStore: Option.some(join(dataDir, "models")),
        cacheRoot: Option.some(join(dataDir, "cache")),
        modelSources: [],
        huggingFaceCaches: resolveHuggingFaceCacheRoots(),
      }),
      host: "127.0.0.1",
      startupTimeout: Duration.seconds(150),
      gracefulShutdownTimeout: Duration.millis(500),
      forceShutdownTimeout: Duration.millis(500),
      outputLimitBytes: 256 * 1024,
      parentPid: process.pid,
    })
  ).pipe(Layer.orDie);

const makeSupervision = () =>
  Layer.scopedDiscard(
    Effect.gen(function* () {
      const icnProcess = yield* IcnProcess;
      const lifecycle = yield* AcnServiceLifecycle;
      yield* icnProcess.unexpectedExit.pipe(
        Effect.catchAll((error) =>
          Effect.logFatal("ICN exited unexpectedly; stopping ACN").pipe(
            Effect.annotateLogs({ cause: error.message }),
            Effect.zipRight(
              lifecycle.beginStopping({ reason: "icn-exited", detail: error.message })
            )
          )
        ),
        Effect.forkScoped
      );
    })
  );

export const makeAcnIcn = (dataDir: string = defaultDataDir()) => {
  const ownership = Layer.effect(
    IcnProcessOwnership,
    Effect.gen(function* () {
      const commandExecutor = yield* CommandExecutor.CommandExecutor
      const fileSystem = yield* FileSystem.FileSystem
      return IcnProcessOwnership.of({
        acquire: ({ instanceId, pid }) => {
          return Effect.gen(function* () {
            const processStartIdentity = yield* readProcessStartIdentity(pid).pipe(
              Effect.mapError((cause) => new IcnLifecycleError({
                operation: "spawn",
                reason: "spawn-failed",
                message: `unable to identify spawned ICN process ${pid}`,
                diagnostic: Option.some(String(cause)),
              })),
              Effect.flatMap(Option.match({
                onNone: () => Effect.fail(new IcnLifecycleError({
                  operation: "spawn",
                  reason: "spawn-failed",
                  message: `spawned ICN process ${pid} was not alive`,
                  diagnostic: Option.none(),
                })),
                onSome: Effect.succeed,
              })),
            )
            const icn: ExactIcnProcess = {
              id: instanceId,
              pid,
              processStartIdentity,
            }
            const observed = yield* readAcnProcessState(dataDir)
            if (
              Option.isNone(observed) ||
              observed.value.mode._tag !== "Assigned" ||
              observed.value.mode.current.id !== ACN_INSTANCE_ID ||
              observed.value.mode.current.pid !== globalThis.process.pid
            ) return yield* new IcnLifecycleError({
              operation: "spawn",
              reason: "spawn-failed",
              message: "ACN lost assignment before ICN ownership acquisition",
              diagnostic: Option.none(),
            })
            yield* applyAcnProcessCommand({
              dataDirectory: dataDir,
              expectedRevision: Option.some(observed.value.revision),
              command: {
                _tag: "RecordIcn",
                acn: observed.value.mode.current,
                icn,
              },
            }).pipe(
              Effect.mapError((cause) => new IcnLifecycleError({
                operation: "spawn",
                reason: "spawn-failed",
                message: "unable to persist ACN ownership of its ICN process",
                diagnostic: Option.some(String(cause)),
              })),
            )
            const release = Effect.gen(function* () {
              const current = yield* readAcnProcessState(dataDir)
              if (
                Option.isNone(current) ||
                current.value.mode._tag !== "Assigned" ||
                current.value.mode.current.id !== ACN_INSTANCE_ID ||
                Option.isNone(current.value.mode.current.ownedIcn) ||
                current.value.mode.current.ownedIcn.value.id !== icn.id ||
                current.value.mode.current.ownedIcn.value.pid !== icn.pid ||
                current.value.mode.current.ownedIcn.value.processStartIdentity !== icn.processStartIdentity
              ) return
              yield* applyAcnProcessCommand({
                dataDirectory: dataDir,
                expectedRevision: Option.some(current.value.revision),
                command: { _tag: "ClearIcn", acn: current.value.mode.current, icn },
              })
            }).pipe(
              Effect.provideService(FileSystem.FileSystem, fileSystem),
              Effect.mapError((cause) => new IcnLifecycleError({
                operation: "shutdown",
                reason: "shutdown-failed",
                message: "unable to release persisted ICN ownership",
                diagnostic: Option.some(String(cause)),
              })),
            )
            return { release }
          }).pipe(
            Effect.mapError((cause) => cause instanceof IcnLifecycleError
              ? cause
              : new IcnLifecycleError({
                  operation: "spawn",
                  reason: "spawn-failed",
                  message: "unable to coordinate ICN ownership",
                  diagnostic: Option.some(String(cause)),
                })),
            Effect.provideService(CommandExecutor.CommandExecutor, commandExecutor),
            Effect.provideService(FileSystem.FileSystem, fileSystem),
          )
        },
      })
    }),
  )
  const preparation = Layer.effect(
    IcnPreparationReporter,
    Effect.gen(function* () {
      const lifecycle = yield* AcnServiceLifecycle;
      const state = yield* Ref.make<{
        readonly plan: Option.Option<AcnInstallationPlan>;
        readonly installationRequired: boolean;
      }>({
        plan: Option.none(),
        installationRequired: false,
      });
      return {
        report: (event) =>
          Effect.gen(function* () {
            switch (event._tag) {
              case "Resolving":
                return yield* lifecycle.reportStarting("Resolving", Option.none());
              case "Planned":
                return yield* Ref.update(state, (current) => ({
                  ...current,
                  plan: Option.some(event.plan),
                }));
              case "InstallationRequired":
                return yield* Ref.update(state, (current) => ({
                  ...current,
                  installationRequired: true,
                }));
              case "Artifact": {
                if (event.event._tag !== "Downloading") return;
                const current = yield* Ref.get(state);
                if (Option.isNone(current.plan)) {
                  return yield* Effect.dieMessage(
                    "ICN artifact download began before its installation plan"
                  );
                }
                yield* Ref.set(state, {
                  ...current,
                  installationRequired: true,
                });
                return yield* lifecycle.reportStarting(
                  {
                    _tag: "Installing",
                    phase: "DownloadingInferenceEngine",
                    plan: current.plan.value,
                  },
                  Option.some(
                    artifactProgress(
                      event.artifact,
                      event.event,
                      current.plan.value
                    )
                  )
                );
              }
              case "Starting": {
                const current = yield* Ref.get(state);
                if (!current.installationRequired) {
                  return yield* lifecycle.reportStarting("Starting", Option.none());
                }
                if (Option.isNone(current.plan)) {
                  return yield* Effect.dieMessage(
                    "ICN installation started without an installation plan"
                  );
                }
                return yield* lifecycle.reportStarting(
                  {
                    _tag: "Installing",
                    phase: "StartingMagnitude",
                    plan: current.plan.value,
                  },
                  Option.none()
                );
              }
              case "PreparingBackend":
                return yield* lifecycle.reportStarting({
                  _tag: "PreparingBackend",
                  backend: event.backend,
                }, Option.none());
            }
          }),
      };
    })
  );
  const process = makeProcess(dataDir).pipe(
    Layer.provide(preparation),
    Layer.provide(ownership),
  );
  const supervisedProcess = Layer.provideMerge(makeSupervision(), process);
  const withClient = Layer.provideMerge(makeIcnClient(), supervisedProcess);
  const withHardware = Layer.provideMerge(makeIcnHardware(), withClient);
  const withCatalog = Layer.provideMerge(makeIcnCatalog(), withHardware);
  const withInstalled = Layer.provideMerge(
    makeIcnInstalledModels(),
    withCatalog
  );
  const withInstances = Layer.provideMerge(IcnInstancesLive, withInstalled);
  const withDownloads = Layer.provideMerge(makeIcnDownloads(), withInstances);
  return withDownloads.pipe(Layer.orDie);
};
