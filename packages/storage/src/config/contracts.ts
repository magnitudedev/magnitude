import { Context, Effect } from 'effect'

import type { ResolvedContextLimitPolicy } from './defaults'
import type { ContextLimitPolicy, MagnitudeConfig, ModelSelection, ProviderOptions, RoleConfig } from '../types'

export interface ConfigStorageShape<TSlot extends string> {
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

  readonly getRoleConfig: (slot: TSlot) => Effect.Effect<RoleConfig | null>
  readonly getRoleConfigs: () => Effect.Effect<Record<TSlot, RoleConfig>>
  readonly getModelSelection: (slot: TSlot) => Effect.Effect<ModelSelection | null>
  readonly setModelSelection: (
    slot: TSlot,
    selection: ModelSelection | null
  ) => Effect.Effect<void>
  readonly getPresets: () => Effect.Effect<Array<{ name: string; models: Record<TSlot, ModelSelection | null> }>>
  readonly savePreset: (
    name: string,
    models: Record<TSlot, ModelSelection | null>
  ) => Effect.Effect<void>
  readonly deletePreset: (name: string) => Effect.Effect<void>
  readonly getProviderOptions: (providerId: string) => Effect.Effect<ProviderOptions | undefined>
  readonly getLocalProviderConfig: () => Effect.Effect<{ baseUrl?: string; modelId?: string } | undefined>
  readonly setLocalProviderConfig: (
    config: { baseUrl?: string; modelId?: string } | undefined
  ) => Effect.Effect<void>
}

export const ConfigStorage = Context.GenericTag<ConfigStorageShape<string>>('ConfigStorage')
export type ConfigStorage = Context.Tag.Identifier<typeof ConfigStorage>

export {
  ConfigStorage as AppConfig,
}
export type AppConfigShape<TSlot extends string> = ConfigStorageShape<TSlot>
