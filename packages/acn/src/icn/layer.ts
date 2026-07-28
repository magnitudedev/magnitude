import { homedir } from "node:os"
import { join, resolve } from "node:path"
import { Duration, Effect, Layer, Option } from "effect"
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
  IcnStorageConfig,
} from "@magnitudedev/icn"
import { ACN_VERSION } from "../version"
import { AcnShutdown } from "../acn-shutdown"
import { resolveHuggingFaceCacheRoots } from "./hugging-face-cache"
const defaultDataDir = () => join(homedir(), ".magnitude")

const binarySource = (dataDir: string) => {
  const explicit = process.env.MAGNITUDE_ICN_PATH?.trim()
  if (explicit) {
    return {
      _tag: "Installation" as const,
      path: explicit,
    }
  }
  if (ACN_VERSION.includes("+dev.")) {
    return {
      _tag: "Installation" as const,
      path: resolve(
        import.meta.dir,
        "../../../../inference/target/development/installation.json",
      ),
    }
  }
  return {
    _tag: "Release" as const,
    version: ACN_VERSION,
    dataDir,
    releaseBaseUrl: (
      process.env.MAGNITUDE_RELEASE_BASE_URL ??
      "https://github.com/magnitudedev/magnitude/releases/download"
    ).replace(/\/+$/, ""),
  }
}

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
      startupTimeout: Duration.seconds(30),
      gracefulShutdownTimeout: Duration.millis(500),
      forceShutdownTimeout: Duration.millis(500),
      outputLimitBytes: 256 * 1024,
      parentPid: process.pid,
    }),
  ).pipe(Layer.orDie)

const makeSupervision = () =>
  Layer.scopedDiscard(
    Effect.gen(function* () {
      const icnProcess = yield* IcnProcess
      const shutdown = yield* AcnShutdown
      yield* icnProcess.unexpectedExit.pipe(
        Effect.catchAll((error) =>
          Effect.logFatal("ICN exited unexpectedly; stopping ACN").pipe(
            Effect.annotateLogs({ cause: error.message }),
            Effect.zipRight(
              shutdown.request({ reason: "icn-exited", detail: error.message }),
            ),
          ),
        ),
        Effect.forkScoped,
      )
    }),
  )

export const makeAcnIcn = (dataDir: string = defaultDataDir()) => {
  const process = makeProcess(dataDir)
  const supervisedProcess = Layer.provideMerge(makeSupervision(), process)
  const withClient = Layer.provideMerge(makeIcnClient(), supervisedProcess)
  const withHardware = Layer.provideMerge(makeIcnHardware(), withClient)
  const withCatalog = Layer.provideMerge(makeIcnCatalog(), withHardware)
  const withInstalled = Layer.provideMerge(makeIcnInstalledModels(), withCatalog)
  const withDownloads = Layer.provideMerge(makeIcnDownloads(), withInstalled)
  return withDownloads.pipe(Layer.orDie)
}
