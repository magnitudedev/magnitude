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
  makeIcnCatalogInstallations,
  makeIcnDiscovery,
  makeIcnModelAssessments,
  makeIcnClient,
  makeIcnProcess,
  makeIcnHardware,
  makeIcnEvents,
  IcnInstancesLive,
  IcnStorageConfig,
  IcnPreparationReporter,
} from "@magnitudedev/icn";
import { ACN_VERSION } from "../version";
import { resolveHuggingFaceCacheRoots } from "./hugging-face-cache";
import { AcnServiceLifecycle } from "../service-lifecycle";

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
        huggingFaceCaches: resolveHuggingFaceCacheRoots(),
      }),
      host: "127.0.0.1",
      startupTimeout: Duration.seconds(150),
      gracefulShutdownTimeout: Duration.millis(500),
      forceShutdownTimeout: Duration.millis(500),
      outputLimitBytes: 256 * 1024,
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
  const process = makeProcess(dataDir).pipe(Layer.provide(preparation));
  const supervisedProcess = Layer.provideMerge(makeSupervision(), process);
  const withClient = Layer.provideMerge(makeIcnClient(), supervisedProcess);
  const withEvents = Layer.provideMerge(makeIcnEvents(), withClient);
  const withHardware = Layer.provideMerge(makeIcnHardware(), withEvents);
  const withCatalog = Layer.provideMerge(makeIcnCatalog(), withHardware);
  const withModels = Layer.provideMerge(makeIcnDiscovery(), withCatalog);
  const withAssessments = Layer.provideMerge(makeIcnModelAssessments(), withModels);
  const withInstallations = Layer.provideMerge(makeIcnCatalogInstallations(), withAssessments);
  return Layer.provideMerge(IcnInstancesLive, withInstallations).pipe(Layer.orDie);
};
