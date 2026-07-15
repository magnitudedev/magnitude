import { Layer } from "effect"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import type * as FileSystem from "@effect/platform/FileSystem"
import type * as HttpClient from "@effect/platform/HttpClient"
import type * as Path from "@effect/platform/Path"
import type {
  LlamaCppDistributionConfig,
  LlamaCppModelStoreConfig,
  LlamaCppRuntimeConfig,
} from "./contracts"
import { LlamaCppDistributionLive } from "./distribution"
import { LlamaCppHostLive } from "./host"
import { LlamaCppModelStoreLive } from "./model-store"
import { LlamaCppRuntimeLive } from "./runtime"

export {
  LlamaCppDistribution,
  LlamaCppDistributionLive,
  type LlamaCppDistributionApi,
} from "./distribution"
export {
  LlamaCppHost,
  LlamaCppHostLive,
  type LlamaCppHostApi,
} from "./host"
export {
  LlamaCppModelStore,
  LlamaCppModelStoreLive,
  type LlamaCppModelStoreApi,
} from "./model-store"
export {
  LlamaCppRuntime,
  LlamaCppRuntimeLive,
  type LlamaCppRuntimeApi,
} from "./runtime"
export * from "./contracts"
export {
  DistributionInspectionError,
  DistributionInstallError,
  LlamaCppHostError,
  LlamaCppModelStoreError,
  LlamaCppRuntimeError,
  LlamaCppEndpointClientError,
} from "./errors"
export { DEFAULT_LLAMACPP_RELEASE } from "./release-manifest"
export {
  MINIMUM_LLMACPP_VERSION,
  RECOMMENDED_LLMACPP_VERSION,
  meetsMinimum,
  parseVersionNumber,
} from "./version"

export interface LlamaCppConfig {
  readonly distribution: LlamaCppDistributionConfig
  readonly modelStore: LlamaCppModelStoreConfig
  readonly runtime: LlamaCppRuntimeConfig
}

type LlamaCppPlatform =
  | FileSystem.FileSystem
  | Path.Path
  | CommandExecutor.CommandExecutor
  | HttpClient.HttpClient

export const LlamaCppLive = (config: LlamaCppConfig) => {
  const distribution = LlamaCppDistributionLive(config.distribution)
  const modelStore = LlamaCppModelStoreLive(config.modelStore)
  const base = Layer.merge(distribution, modelStore)
  const withHost = Layer.provideMerge(LlamaCppHostLive, base)
  return Layer.provideMerge(LlamaCppRuntimeLive(config.runtime), withHost) satisfies Layer.Layer<
    import("./distribution").LlamaCppDistribution
      | import("./host").LlamaCppHost
      | import("./model-store").LlamaCppModelStore
      | import("./runtime").LlamaCppRuntime,
    never,
    LlamaCppPlatform
  >
}
