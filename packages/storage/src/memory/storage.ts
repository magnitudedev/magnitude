import {
  pathExists,
  readTextFile,
  writeTextFile,
} from '../io'
import type { ProjectStoragePaths } from '../paths'

export async function ensureMemoryFile(
  paths: ProjectStoragePaths,
  template = ''
): Promise<string> {
  if (await pathExists(paths.memoryFile)) {
    return paths.memoryFile
  }

  await writeTextFile(paths.memoryFile, template)
  return paths.memoryFile
}

export async function readMemory(
  paths: ProjectStoragePaths
): Promise<string> {
  return readTextFile(paths.memoryFile)
}

export async function writeMemory(
  paths: ProjectStoragePaths,
  content: string
): Promise<void> {
  await writeTextFile(paths.memoryFile, content)
}