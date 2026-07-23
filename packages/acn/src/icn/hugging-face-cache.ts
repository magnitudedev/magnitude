import { statSync } from "node:fs"
import { homedir } from "node:os"
import { isAbsolute, join, resolve } from "node:path"

export interface HuggingFaceCacheResolutionOptions {
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly homeDirectory?: string
  readonly isDirectory?: (path: string) => boolean
}

const existingDirectory = (path: string): boolean => {
  try {
    return statSync(path).isDirectory()
  } catch {
    return false
  }
}

const expandPath = (path: string, homeDirectory: string): string => {
  const expanded = path === "~"
    ? homeDirectory
    : path.startsWith("~/") || path.startsWith("~\\")
      ? join(homeDirectory, path.slice(2))
      : path
  return isAbsolute(expanded) ? expanded : resolve(expanded)
}

export const resolveHuggingFaceCacheRoots = (
  options: HuggingFaceCacheResolutionOptions = {},
): readonly string[] => {
  const env = options.env ?? process.env
  const homeDirectory = options.homeDirectory ?? homedir()
  const configuredHubCache = env.HF_HUB_CACHE?.trim()
    || env.HUGGINGFACE_HUB_CACHE?.trim()
  const configuredHome = env.HF_HOME?.trim()
  const configuredXdgHome = env.XDG_CACHE_HOME?.trim()
  const candidate = configuredHubCache
    ?? (configuredHome
      ? join(configuredHome, "hub")
      : configuredXdgHome
        ? join(configuredXdgHome, "huggingface", "hub")
        : join(homeDirectory, ".cache", "huggingface", "hub"))
  const root = expandPath(candidate, homeDirectory)
  return (options.isDirectory ?? existingDirectory)(root) ? [root] : []
}
