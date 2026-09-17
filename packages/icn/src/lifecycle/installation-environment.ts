import { win32 } from "node:path"

/** Windows' loader needs extended paths even when the filesystem accepted the installation. */
export const installationNativePath = (path: string, platform: NodeJS.Platform = process.platform): string =>
  platform === "win32" ? win32.toNamespacedPath(path) : path

export const installationLoaderEnvironment = (
  runtime: string,
  platform: NodeJS.Platform = process.platform,
  inheritedPath: string | undefined = process.env.PATH,
): Readonly<Record<string, string>> => {
  if (platform === "win32") {
    const nativeRuntime = installationNativePath(runtime, platform)
    return {
      PATH: inheritedPath ? `${nativeRuntime};${inheritedPath}` : nativeRuntime,
    }
  }
  return platform === "darwin"
    ? { DYLD_LIBRARY_PATH: "" }
    : { LD_LIBRARY_PATH: "" }
}
