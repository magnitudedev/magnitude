import { Effect, Layer } from 'effect'

import { resolveContextLimitPolicy } from './defaults'
import { ConfigStorage, type ConfigStorageShape } from './contracts'
import { loadConfig, saveConfig, updateConfig } from './storage'
import { GlobalStorage, type GlobalStorageShape } from '../services'
import type { ContextLimitPolicy, MagnitudeConfig, ModelSelection, ProviderOptions, RoleConfig } from '../types'

const makeConfigStorageShape = <TSlot extends string>(
  globalStorage: GlobalStorageShape
): ConfigStorageShape<TSlot> => ({
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

  getRoleConfig: (slot: TSlot) =>
    Effect.map(
      Effect.promise(() => loadConfig(globalStorage.paths)),
      (config) => config.roles[slot] ?? null,
    ),

  getRoleConfigs: () =>
    Effect.map(
      Effect.promise(() => loadConfig(globalStorage.paths)),
      (config) => config.roles as Record<TSlot, RoleConfig>,
    ),

  getModelSelection: (slot: TSlot) =>
    Effect.map(
      Effect.promise(() => loadConfig(globalStorage.paths)),
      (config) => config.roles[slot]?.model ?? null,
    ),

  setModelSelection: (slot: TSlot, selection: ModelSelection | null) =>
    Effect.as(
      Effect.promise(() =>
        updateConfig(globalStorage.paths, (config) => ({
          ...config,
          roles: {
            ...config.roles,
            [slot]: { ...(config.roles[slot] ?? {}), model: selection },
          },
        })),
      ),
      undefined,
    ),

  getProviderOptions: (providerId: string) =>
    Effect.map(
      Effect.promise(() => loadConfig(globalStorage.paths)),
      (config): ProviderOptions | undefined => config.providers?.[providerId],
    ),

  getLocalProviderConfig: () =>
    Effect.map(Effect.promise(() => loadConfig(globalStorage.paths)), (config) => {
      const local = config.providers?.['local']
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
        providers: localConfig
          ? { ...config.providers, local: localConfig }
          : (() => {
              const { local, ...rest } = config.providers ?? {}
              return rest
            })(),
      }))
    }),
})

export const ConfigStorageLive = Layer.effect(
  ConfigStorage,
  Effect.gen(function* () {
    const globalStorage = yield* GlobalStorage
    return ConfigStorage.of(
      makeConfigStorageShape<string>(globalStorage) as ConfigStorageShape<string>
    )
  })
)
