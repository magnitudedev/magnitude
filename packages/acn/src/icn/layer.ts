import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { Duration, Effect, Layer, Option, Ref } from "effect";
import {
  type AcnInstallationPlan,
  type AcnStartupProgress,
} from "@magnitudedev/acn-protocol";
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
} from "@magnitudedev/icn";
import { ACN_VERSION } from "../version";
import { AcnShutdown } from "../acn-shutdown";
import { resolveHuggingFaceCacheRoots } from "./hugging-face-cache";
import { AcnStartupState } from "../startup-state";

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
          "model_fit",
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
      const shutdown = yield* AcnShutdown;
      yield* icnProcess.unexpectedExit.pipe(
        Effect.catchAll((error) =>
          Effect.logFatal("ICN exited unexpectedly; stopping ACN").pipe(
            Effect.annotateLogs({ cause: error.message }),
            Effect.zipRight(
              shutdown.request({ reason: "icn-exited", detail: error.message })
            )
          )
        ),
        Effect.forkScoped
      );
    })
  );

export const makeAcnIcn = (dataDir: string = defaultDataDir()) => {
  const preparation = Layer.effect(
    IcnPreparationReporter,
    Effect.gen(function* () {
      const startup = yield* AcnStartupState;
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
                return yield* startup.starting("Resolving", Option.none());
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
                return yield* startup.installing(
                  "DownloadingInferenceEngine",
                  current.plan.value,
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
                  return yield* startup.starting("Starting", Option.none());
                }
                if (Option.isNone(current.plan)) {
                  return yield* Effect.dieMessage(
                    "ICN installation started without an installation plan"
                  );
                }
                return yield* startup.installing(
                  "StartingMagnitude",
                  current.plan.value,
                  Option.none()
                );
              }
              case "PreparingBackend":
                return yield* startup.starting({
                  _tag: "PreparingBackend",
                  backend: event.backend,
                }, Option.none());
            }
          }),
      };
    })
  );
  const process = makeProcess(dataDir).pipe(Layer.provide(preparation));
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
