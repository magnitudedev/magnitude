import { Context, Effect } from 'effect'

import type { ResolvedContextLimitPolicy } from './defaults'
import type { ContextLimitPolicy, MagnitudeConfig, ModelSelection, ProviderOptions } from '../types'

export interface ConfigStorageShape {
  readonly load: () => Effect.Effect<MagnitudeConfig>
  readonly save: (config: MagnitudeConfig) => Effect.Effect<void>
  readonly update: (
    f: (config: MagnitudeConfig) => MagnitudeConfig
  ) => Effect.Effect<MagnitudeConfig>

  readonly getContextLimitPolicy: () => Effect.Effect<ResolvedContextLimitPolicy>
  readonly setContextLimitPolicy: (
    policy: ContextLimitPolicy
  ) => Effect.Effect<void>

  readonly getSetupComplete: () => Effect.Effect<boolean>
  readonly setSetupComplete: (value: boolean) => Effect.Effect<void>
  readonly getTelemetryEnabled: () => Effect.Effect<boolean>
  readonly setTelemetryEnabled: (value: boolean) => Effect.Effect<void>
  readonly getMemoryEnabled: () => Effect.Effect<boolean>

  readonly getModelSelection: (
    slot: 'primary' | 'secondary' | 'browser'
  ) => Effect.Effect<ModelSelection | null>
  readonly setModelSelection: (
    slot: 'primary' | 'secondary' | 'browser',
    selection: ModelSelection | null
  ) => Effect.Effect<void>
  readonly getProviderOptions: (providerId: string) => Effect.Effect<ProviderOptions | undefined>
  readonly getLocalProviderConfig: () => Effect.Effect<{ baseUrl?: string; modelId?: string } | undefined>
  readonly setLocalProviderConfig: (
    config: { baseUrl?: string; modelId?: string } | undefined
  ) => Effect.Effect<void>
}

export class ConfigStorage extends Context.Tag('ConfigStorage')<
  ConfigStorage,
  ConfigStorageShape
>() {}

export {
  ConfigStorage as AppConfig,
  type ConfigStorageShape as AppConfigShape,
}