import { Effect, Layer } from 'effect'

import { resolveContextLimitPolicy } from './defaults'
import { ConfigStorage } from './contracts'
import { loadConfig, saveConfig, updateConfig } from './storage'
import { GlobalStorage } from '../services'
import type { ContextLimitPolicy, MagnitudeConfig, ModelSelection, ProviderOptions } from '../types'

export const ConfigStorageLive = Layer.effect(
  ConfigStorage,
  Effect.gen(function* () {
    const globalStorage = yield* GlobalStorage

    return ConfigStorage.of({
      load: () => Effect.promise(() => loadConfig(globalStorage.paths)),
      save: (config: MagnitudeConfig) =>
        Effect.promise(() => saveConfig(globalStorage.paths, config)),
      update: (f: (config: MagnitudeConfig) => MagnitudeConfig) =>
        Effect.promise(() => updateConfig(globalStorage.paths, f)),

      getContextLimitPolicy: () =>
        Effect.promise(async () =>
          resolveContextLimitPolicy(await loadConfig(globalStorage.paths))
        ),

      setContextLimitPolicy: (policy: ContextLimitPolicy) =>
        Effect.promise(async () => {
          await updateConfig(globalStorage.paths, (config) => ({
            ...config,
            contextLimits: {
              ...(config.contextLimits ?? {}),
              ...policy,
            },
          }))
        }),

      getSetupComplete: () =>
        Effect.promise(async () => {
          const config = await loadConfig(globalStorage.paths)
          return config.setupComplete === true
        }),

      setSetupComplete: (value: boolean) =>
        Effect.promise(async () => {
          await updateConfig(globalStorage.paths, (config) => ({
            ...config,
            setupComplete: value,
          }))
        }),

      getTelemetryEnabled: () =>
        Effect.promise(async () => {
          const config = await loadConfig(globalStorage.paths)
          return config.telemetry !== false
        }),

      setTelemetryEnabled: (value: boolean) =>
        Effect.promise(async () => {
          await updateConfig(globalStorage.paths, (config) => ({
            ...config,
            telemetry: value,
          }))
        }),

      getMemoryEnabled: () =>
        Effect.promise(async () => {
          const config = await loadConfig(globalStorage.paths)
          return config.memory !== false
        }),

      getModelSelection: (slot: 'primary' | 'secondary' | 'browser') =>
        Effect.map(Effect.promise(() => loadConfig(globalStorage.paths)), (config) => {
          if (slot === 'primary') return config.primaryModel
          if (slot === 'secondary') return config.secondaryModel
          return config.browserModel
        }),

      setModelSelection: (
        slot: 'primary' | 'secondary' | 'browser',
        selection: ModelSelection | null,
      ) =>
        Effect.as(
          Effect.promise(() =>
            updateConfig(globalStorage.paths, (config) => {
              if (slot === 'primary') return { ...config, primaryModel: selection }
              if (slot === 'secondary') return { ...config, secondaryModel: selection }
              return { ...config, browserModel: selection }
            }),
          ),
          undefined,
        ),

      getProviderOptions: (providerId: string) =>
        Effect.map(
          Effect.promise(() => loadConfig(globalStorage.paths)),
          (config): ProviderOptions | undefined => config.providerOptions?.[providerId],
        ),

      getLocalProviderConfig: () =>
        Effect.map(Effect.promise(() => loadConfig(globalStorage.paths)), (config) => {
          const local = config.providerOptions?.['local']
          if (!local) return undefined
          return {
            baseUrl: local.baseUrl,
            modelId: local.modelId,
          }
        }),

      setLocalProviderConfig: (localConfig) =>
        Effect.promise(async () => {
          await updateConfig(globalStorage.paths, (config) => ({
            ...config,
            providerOptions: localConfig
              ? { ...config.providerOptions, local: localConfig }
              : (() => {
                  const { local, ...rest } = config.providerOptions ?? {}
                  return rest
                })(),
          }))
        }),
    })
  })
)