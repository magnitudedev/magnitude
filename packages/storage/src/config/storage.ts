import { readJsonFileWithSchema, writeJsonFile } from '../io'
import {
  defaultGlobalStorageRoot,
  makeGlobalStoragePaths,
  type GlobalStoragePaths,
} from '../paths'
import {
  MagnitudeConfigSchema,
  type MagnitudeConfig,
} from '../types'

export const DEFAULT_CONFIG: MagnitudeConfig = {
  primaryModel: null,
  secondaryModel: null,
  browserModel: null,
}

function getDefaultPaths(): GlobalStoragePaths {
  return makeGlobalStoragePaths(defaultGlobalStorageRoot())
}

export async function loadConfig(
  paths: GlobalStoragePaths = getDefaultPaths()
): Promise<MagnitudeConfig> {
  return readJsonFileWithSchema(paths.configFile, MagnitudeConfigSchema, {
    fallback: DEFAULT_CONFIG,
  }) as Promise<MagnitudeConfig>
}

export async function saveConfig(
  paths: GlobalStoragePaths,
  config: MagnitudeConfig
): Promise<void> {
  await writeJsonFile(paths.configFile, config)
}

export async function updateConfig(
  paths: GlobalStoragePaths,
  fn: (config: MagnitudeConfig) => MagnitudeConfig | Promise<MagnitudeConfig>
): Promise<MagnitudeConfig> {
  const nextConfig = await fn(await loadConfig(paths))
  await saveConfig(paths, nextConfig)
  return nextConfig
}

export async function setPrimarySelection(
  paths: GlobalStoragePaths,
  providerId: string,
  modelId: string
): Promise<void> {
  await updateConfig(paths, (config) => ({
    ...config,
    primaryModel: { providerId, modelId },
  }))
}

export async function setBrowserSelection(
  paths: GlobalStoragePaths,
  providerId: string,
  modelId: string
): Promise<void> {
  await updateConfig(paths, (config) => ({
    ...config,
    browserModel: { providerId, modelId },
  }))
}