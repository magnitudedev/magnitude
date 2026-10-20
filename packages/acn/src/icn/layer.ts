import { ProcessGroupController } from "@magnitudedev/utils/process-groups";
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native";
import { FetchHttpClient, FileSystem } from "@effect/platform";
import { ModelAssessmentsSnapshot } from "@magnitudedev/icn-protocol/schemas";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { nativeWindowsJobOwnerLayer, nativeWindowsPrivatePipesLayer } from "@magnitudedev/utils/windows-native";
import { WindowsIcnChildSpawner } from "./windows-child";
import { Duration, Effect, Layer, Option, Ref, Schema } from "effect";
import type { LocalModelsConfiguration } from "@magnitudedev/icn-protocol/schemas";
import { EMPTY_CUSTOM_MODELS_CONFIG, MagnitudeStorage, resolveUsableModelStoreLocation } from "@magnitudedev/storage";
import {
  type AcnInstallationPlan,
  type AcnStartupProgress,
} from "@magnitudedev/acn-protocol";
import type { ArtifactInstallationEvent } from "@magnitudedev/release";
import {
  IcnBinaryResolutionConfig,
  IcnLifecycleConfig,
  IcnProcess,
  IcnModelAssessments,
  makeIcnCatalog,
  makeIcnCatalogInstallations,
  makeIcnLocalModels,
  makeIcnDownloads,
  IcnCandidatesLive,
  makeIcnModelAssessments,
  makeIcnClient,
  makeIcnHuggingFaceCredential,
  makeIcnProcess,
  UnixIcnChildSpawner,
  makeIcnHardware,
  makeIcnEvents,
  IcnInstancesLive,
  IcnStorageConfig,
  IcnPreparationReporter,
} from "@magnitudedev/icn";
import { ACN_VERSION } from "../version";
import { hostDiscoveryEnvironment, resolveAutomaticLocations } from "./discovery-roots";
import { localModelsConfiguration } from "../custom-models";
import { HuggingFaceCredentialDeliveryLive } from "../hugging-face-credential-delivery";
import { AcnServiceLifecycle } from "../service-lifecycle";

const artifactProgress = (
  event: Extract<ArtifactInstallationEvent, { readonly _tag: "Downloading" }>,
  plan: AcnInstallationPlan
): AcnStartupProgress => ({
  completed: Math.min(plan.inferenceEngineBytes, event.progress.acceptedBytes),
  totalBytes: plan.inferenceEngineBytes,
  unit: "Bytes",
  attempt: Option.some(event.progress.attempt),
});
const defaultDataDir = () => join(homedir(), ".magnitude");
// ICN bounds its own work (a stalled Hub transfer fails there), so its calls carry no client deadline.
// Bun's fetch otherwise ends every request after five minutes, and a repository inspection that
// outlasts that on a slow connection would be reported as ICN being unreachable.
const icnFetchOptions: globalThis.RequestInit & { readonly timeout: false } = { timeout: false };
export const icnHttpClient = FetchHttpClient.layer.pipe(
  Layer.provide(Layer.succeed(FetchHttpClient.RequestInit, icnFetchOptions)),
);
const AcceptanceAssessmentDiagnostics = Schema.Struct({
  assessments: ModelAssessmentsSnapshot,
  inferenceDiagnosticTail: Schema.String,
});

const acceptanceAssessmentDiagnostics = Layer.scopedDiscard(Effect.gen(function* () {
  const path = process.env.MAGNITUDE_ACCEPTANCE_ASSESSMENT_DIAGNOSTICS;
  if (!path) return;
  const fs = yield* FileSystem.FileSystem;
  const assessments = yield* IcnModelAssessments;
  const icnProcess = yield* IcnProcess;
  yield* Effect.gen(function* () {
    for (;;) {
      yield* Effect.gen(function* () {
        const snapshot = (yield* assessments.get).state;
        const inferenceDiagnosticTail = yield* icnProcess.diagnosticTail;
        const report = yield* Schema.encode(Schema.parseJson(AcceptanceAssessmentDiagnostics))({
          assessments: snapshot,
          inferenceDiagnosticTail,
        });
        yield* fs.writeFileString(`${path}.pending`, report);
        yield* fs.rename(`${path}.pending`, path);
      }).pipe(Effect.catchAll(() => Effect.void));
      yield* Effect.sleep("5 seconds");
    }
  }).pipe(Effect.forkScoped);
}));

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

/**
 * The engine receives its store root once, at spawn. The configured `modelsDirectory` is read
 * here so a Settings change or a hand edit of config.json applies at the next service start.
 */
const resolveModelStore = (dataDir: string) =>
  Effect.gen(function* () {
    const storage = yield* MagnitudeStorage;
    const configured = yield* storage.config.load().pipe(
      Effect.map((config) => config.modelsDirectory),
      Effect.tapError((error) =>
        Effect.logWarning("Could not read modelsDirectory from config.json; using the default model store").pipe(
          Effect.annotateLogs({ cause: error.message })
        )
      ),
      Effect.orElseSucceed(() => Option.none<string>())
    );
    const location = yield* resolveUsableModelStoreLocation(dataDir, configured);
    if (Option.isSome(location.warning)) yield* Effect.logWarning(location.warning.value);
    yield* Effect.logInfo("Model store resolved").pipe(
      Effect.annotateLogs({ root: location.root, source: location.source })
    );
    return location.root;
  });

/**
 * The local-models snapshot ICN scans first, from the saved choices as they stand at spawn, chosen
 * drafters included with their persisted methods.
 */
const resolveLocalModelsSnapshot = Effect.gen(function* () {
  const storage = yield* MagnitudeStorage;
  const config = yield* storage.config.load().pipe(
    Effect.map((loaded) => Option.getOrElse(loaded.models, () => EMPTY_CUSTOM_MODELS_CONFIG)),
    Effect.tapError((error) =>
      Effect.logWarning("Could not read model choices from config.json; scanning only the automatic locations").pipe(
        Effect.annotateLogs({ cause: error.message })
      )
    ),
    Effect.orElseSucceed(() => EMPTY_CUSTOM_MODELS_CONFIG)
  );
  const locations = yield* resolveAutomaticLocations(hostDiscoveryEnvironment());
  return localModelsConfiguration(config, locations);
});

const makeProcess = (dataDir: string, modelStore: string, localModels: LocalModelsConfiguration) =>
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
        modelStore: Option.some(modelStore),
        cacheRoot: Option.some(join(dataDir, "cache")),
        localModelsConfigurationFile: join(dataDir, "cache", "local-models-configuration.json"),
        localModels,
      }),
      host: "127.0.0.1",
      startupTimeout: Duration.seconds(150),
      gracefulShutdownTimeout: Duration.millis(500),
      forceShutdownTimeout: Duration.millis(500),
      outputLimitBytes: 256 * 1024,
      logFile: Option.some(join(dataDir, "logs", "inference.log")),
    })
  ).pipe(Layer.provide(process.platform === "win32"
    ? WindowsIcnChildSpawner.pipe(Layer.provide(Layer.merge(
      nativeWindowsJobOwnerLayer(process.env.MAGNITUDE_NATIVE_HOST ?? join(dirname(process.execPath), "desktop-host.node")),
      nativeWindowsPrivatePipesLayer(process.env.MAGNITUDE_NATIVE_HOST ?? join(dirname(process.execPath), "desktop-host.node")),
    )))
    : UnixIcnChildSpawner.pipe(Layer.provide(Layer.succeed(ProcessGroupController, ProcessGroupControllerLive)))));

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
                    artifactProgress(event.event, current.plan.value)
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
  const process = Layer.unwrapEffect(
    Effect.all([resolveModelStore(dataDir), resolveLocalModelsSnapshot]).pipe(
      Effect.map(([modelStore, localModels]) =>
        makeProcess(dataDir, modelStore, localModels).pipe(Layer.provide(preparation)))
    )
  );
  const supervisedProcess = Layer.provideMerge(makeSupervision(), process);
  const withClient = Layer.provideMerge(makeIcnClient().pipe(Layer.provide(icnHttpClient)), supervisedProcess);
  const withHubCredential = Layer.provideMerge(
    HuggingFaceCredentialDeliveryLive.pipe(Layer.provideMerge(makeIcnHuggingFaceCredential())),
    withClient,
  );
  const withEvents = Layer.provideMerge(makeIcnEvents(), withHubCredential);
  const withHardware = Layer.provideMerge(makeIcnHardware(), withEvents);
  const withCatalog = Layer.provideMerge(makeIcnCatalog(), withHardware);
  const withLocalModels = Layer.provideMerge(makeIcnLocalModels(), withCatalog);
  const withDownloads = Layer.provideMerge(makeIcnDownloads(), withLocalModels);
  const withModels = Layer.provideMerge(IcnCandidatesLive, withDownloads);
  const withAssessments = Layer.provideMerge(makeIcnModelAssessments(), withModels);
  const withDiagnostics = globalThis.process.env.MAGNITUDE_ACCEPTANCE_ASSESSMENT_DIAGNOSTICS
    ? Layer.provideMerge(acceptanceAssessmentDiagnostics, withAssessments)
    : withAssessments;
  const withInstallations = Layer.provideMerge(makeIcnCatalogInstallations(), withDiagnostics);
  return Layer.provideMerge(IcnInstancesLive, withInstallations);
};
