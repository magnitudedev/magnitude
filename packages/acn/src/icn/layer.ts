import { homedir } from "node:os"
import { join, resolve } from "node:path"
import { Duration, Effect, Layer, Option } from "effect"
import {
  IcnBinaryResolutionConfig,
  IcnLifecycleConfig,
  IcnLifecycle,
  makeIcn,
  makeIcnHardware,
  makeIcnInventory,
  makeIcnProvider,
  makeIcnRecipes,
  IcnStorageConfig,
} from "@magnitudedev/icn"
import { ACN_VERSION } from "../version"

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
        resolve(import.meta.dir, "../../../../inference/target/debug/magnitude-icn"),
        resolve(import.meta.dir, "../../../../inference/target/release/magnitude-icn"),
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

const makeLifecycle = () => makeIcn(new IcnLifecycleConfig({
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
      "model_load_control",
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

const makeSupervision = () => Layer.scopedDiscard(Effect.gen(function* () {
  const icn = yield* IcnLifecycle
  yield* icn.unexpectedExit.pipe(
    Effect.catchAll((cause) => Effect.logFatal("ICN exited unexpectedly; stopping ACN").pipe(
      Effect.annotateLogs({ cause: cause.message }),
      // BunRuntime translates SIGTERM into root interruption, so all scoped ACN and ICN
      // finalizers run before the process exits.
      Effect.zipRight(Effect.sync(() => process.kill(process.pid, "SIGTERM"))),
    )),
    Effect.forkScoped,
  )
}))

export const makeAcnIcn = () => {
  const lifecycle = makeLifecycle()
  const supervisedLifecycle = Layer.provideMerge(makeSupervision(), lifecycle)
  const withHardware = Layer.provideMerge(makeIcnHardware(), supervisedLifecycle)
  const withInventory = Layer.provideMerge(makeIcnInventory(), withHardware)
  const withRecipes = Layer.provideMerge(makeIcnRecipes(), withInventory)
  return Layer.provideMerge(makeIcnProvider(), withRecipes).pipe(Layer.orDie)
}
