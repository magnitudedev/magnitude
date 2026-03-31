import { readJsonFileWithSchema, writeJsonFile } from '../io'
import {
  defaultGlobalStorageRoot,
  makeGlobalStoragePaths,
  type GlobalStoragePaths,
} from '../paths'
import { Schema } from 'effect'
import {
  MagnitudeConfigSchema,
  type MagnitudeConfig,
} from '../types'

function getDefaultPaths(): GlobalStoragePaths {
  return makeGlobalStoragePaths(defaultGlobalStorageRoot())
}

export async function loadConfig(
  paths: GlobalStoragePaths = getDefaultPaths()
): Promise<MagnitudeConfig> {
  return readJsonFileWithSchema(paths.configFile, MagnitudeConfigSchema, {
    fallback: Schema.decodeUnknownSync(MagnitudeConfigSchema)({}),
  })
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
