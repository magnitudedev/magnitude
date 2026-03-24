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
  roles: {},
}

function getDefaultPaths(): GlobalStoragePaths {
  return makeGlobalStoragePaths(defaultGlobalStorageRoot())
}

export async function loadConfig(
  paths: GlobalStoragePaths = getDefaultPaths()
): Promise<MagnitudeConfig> {
  const config = (await readJsonFileWithSchema(paths.configFile, MagnitudeConfigSchema, {
    fallback: DEFAULT_CONFIG,
  })) as MagnitudeConfig

  return {
    ...config,
    roles: config.roles ?? {},
  }
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
