import type { PlatformError } from '@effect/platform/Error'
import { Context, Effect } from 'effect'

import type { JsonError } from '../io/storage'
import type { ResolvedContextLimitPolicy } from '../types/config'
import type { ContextLimitPolicy, MagnitudeConfig, ModelConfig, OnboardingConfig, SlotId, SlotModelConfig } from '../types'

export interface ConfigStorageShape {
  readonly load: () => Effect.Effect<MagnitudeConfig, PlatformError | JsonError>
  readonly save: (config: MagnitudeConfig) => Effect.Effect<void, PlatformError | JsonError>
  readonly update: (
    f: (config: MagnitudeConfig) => MagnitudeConfig
  ) => Effect.Effect<MagnitudeConfig, PlatformError | JsonError>

  readonly getContextLimitPolicy: () => Effect.Effect<ResolvedContextLimitPolicy, PlatformError | JsonError>
  readonly setContextLimitPolicy: (
    policy: ContextLimitPolicy
  ) => Effect.Effect<void, PlatformError | JsonError>

  readonly getModelConfig: () => Effect.Effect<ModelConfig | null, PlatformError | JsonError>
  readonly updateModelConfig: (
    slots: Partial<Record<SlotId, SlotModelConfig>>
  ) => Effect.Effect<void, PlatformError | JsonError>

  readonly getOnboardingConfig: () => Effect.Effect<OnboardingConfig | null, PlatformError | JsonError>
  readonly completeCliModelSetupOnboarding: (
    version: number,
    completedAt: string
  ) => Effect.Effect<void, PlatformError | JsonError>
}

export const ConfigStorage = Context.GenericTag<ConfigStorageShape>('ConfigStorage')
export type ConfigStorage = Context.Tag.Identifier<typeof ConfigStorage>

export {
  ConfigStorage as AppConfig,
}
export type AppConfigShape = ConfigStorageShape
