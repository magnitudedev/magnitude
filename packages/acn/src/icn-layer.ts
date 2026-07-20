import { homedir } from "node:os"
import { join, resolve } from "node:path"
import { Duration, Effect, Layer, Option } from "effect"
import {
  IcnBinaryResolutionConfig,
  IcnLifecycleConfig,
  IcnLifecycle,
  IcnLive,
  IcnStorageConfig,
} from "@magnitudedev/icn"
import { ACN_VERSION } from "./version"

const platformKey = (): string => {
  if (process.platform === "darwin" && process.arch === "arm64") return "darwin-arm64"
  if (process.platform === "darwin" && process.arch === "x64") return "darwin-x64"
  if (process.platform === "linux" && process.arch === "x64") return "linux-x64"
  if (process.platform === "linux" && process.arch === "arm64") return "linux-arm64"
  if (process.platform === "win32" && process.arch === "x64") return "windows-x64"
  throw new Error(`Unsupported ICN platform: ${process.platform} ${process.arch}`)
}

const dataDir = join(homedir(), ".magnitude")

const binarySource = () => {
  const explicit = process.env.MAGNITUDE_ICN_PATH?.trim()
  if (explicit) return { _tag: "Explicit" as const, path: explicit }
  if (ACN_VERSION.includes("+dev.")) {
    return {
      _tag: "DevelopmentSearch" as const,
      candidates: [
        resolve(import.meta.dir, "../../../inference/target/debug/magnitude-icn"),
        resolve(import.meta.dir, "../../../inference/target/release/magnitude-icn"),
        resolve(process.cwd(), "inference/target/debug/magnitude-icn"),
        resolve(process.cwd(), "inference/target/release/magnitude-icn"),
      ] as const,
    }
  }
  return {
    _tag: "Release" as const,
    version: ACN_VERSION,
    platformKey: platformKey(),
    dataDir,
    releaseBaseUrl: (process.env.MAGNITUDE_RELEASE_BASE_URL
      ?? "https://github.com/magnitudedev/magnitude/releases/download").replace(/\/+$/, ""),
  }
}

const lifecycle = IcnLive(new IcnLifecycleConfig({
  binary: new IcnBinaryResolutionConfig({
    source: binarySource(),
    supportedApiVersion: 1,
    expectedNativeBuild: Option.none(),
    expectedTarget: Option.none(),
    requiredCapabilities: [
      "hardware",
      "model_inventory",
      "model_preview",
      "model_download",
      "runtime_model_control",
      "chat_streaming",
    ],
    allowBuildMismatch: false,
    probeTimeout: Duration.seconds(10),
    downloadTimeout: Duration.minutes(10),
  }),
  storage: new IcnStorageConfig({
    modelStore: Option.some(join(dataDir, "models")),
    modelSources: [],
    huggingFaceCaches: [],
  }),
  host: "127.0.0.1",
  startupTimeout: Duration.seconds(30),
  gracefulShutdownTimeout: Duration.seconds(15),
  forceShutdownTimeout: Duration.seconds(5),
  outputLimitBytes: 256 * 1024,
  parentPid: process.pid,
})).pipe(Layer.orDie)

const supervision = Layer.scopedDiscard(Effect.gen(function* () {
  const icn = yield* IcnLifecycle
  yield* icn.unexpectedExit.pipe(
    Effect.catchAll((cause) => Effect.logFatal("ICN exited unexpectedly; terminating ACN").pipe(
      Effect.annotateLogs({ cause: cause.message }),
      Effect.zipRight(Effect.sync(() => process.exit(1))),
    )),
    Effect.forkScoped,
  )
}))

export const AcnIcnLive = Layer.provideMerge(supervision, lifecycle)
