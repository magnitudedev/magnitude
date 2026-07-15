import { homedir } from "node:os"
import { join } from "node:path"

export const llamacppDataDir = (): string =>
  join(homedir(), ".magnitude", "bin", "llamacpp")

export const cachedBinaryDir = (buildNumber: number): string =>
  join(llamacppDataDir(), `llama-b${buildNumber}`)

export const cachedBinaryPath = (buildNumber: number): string =>
  join(cachedBinaryDir(buildNumber), "llama-server")

export const versionMarkerPath = (): string =>
  join(llamacppDataDir(), "llama-server.version")

export const downloadTmpDir = (): string =>
  join(homedir(), ".magnitude", "downloads", "llamacpp")

export const presetDir = (): string =>
  join(homedir(), ".magnitude", "llamacpp", "presets")
