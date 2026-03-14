import {
  readJsonFile,
  removeFileIfExists,
  writeSecureJsonFile,
} from '../io'
import {
  defaultGlobalStorageRoot,
  makeGlobalStoragePaths,
  type GlobalStoragePaths,
} from '../paths'
import {
  isValidAuthInfo,
  type AuthInfo,
} from '../types'

function normalizeAuthData(data: unknown): Record<string, AuthInfo> {
  if (typeof data !== 'object' || data === null) {
    return {}
  }

  const result: Record<string, AuthInfo> = {}
  const isAuthInfo = isValidAuthInfo

  for (const [key, value] of Object.entries(data)) {
    if (isAuthInfo(value)) {
      result[key] = value
    }
  }

  return result
}

function getDefaultPaths(): GlobalStoragePaths {
  return makeGlobalStoragePaths(defaultGlobalStorageRoot())
}

export const AUTH_PATH = getDefaultPaths().authFile

export async function loadAuth(
  paths: GlobalStoragePaths = getDefaultPaths()
): Promise<Record<string, AuthInfo>> {
  const raw = await readJsonFile<unknown>(paths.authFile, {
    fallback: {},
  })

  return normalizeAuthData(raw)
}

export async function getAuth(
  paths: GlobalStoragePaths,
  providerId: string
): Promise<AuthInfo | undefined> {
  return (await loadAuth(paths))[providerId]
}

export async function setAuth(
  paths: GlobalStoragePaths,
  providerId: string,
  info: AuthInfo
): Promise<void> {
  const data = await loadAuth(paths)
  data[providerId] = info
  await writeSecureJsonFile(paths.authFile, data)
}

export async function removeAuth(
  paths: GlobalStoragePaths,
  providerId: string
): Promise<void> {
  const data = await loadAuth(paths)
  delete data[providerId]

  if (Object.keys(data).length === 0) {
    await removeFileIfExists(paths.authFile)
    return
  }

  await writeSecureJsonFile(paths.authFile, data)
}