export const installationLoaderEnvironment = (
  runtime: string,
  platform: NodeJS.Platform = process.platform,
  inheritedPath: string | undefined = process.env.PATH,
): Readonly<Record<string, string>> => {
  if (platform === "win32") {
    return {
      PATH: inheritedPath ? `${runtime};${inheritedPath}` : runtime,
    }
  }
  return platform === "darwin"
    ? { DYLD_LIBRARY_PATH: "" }
    : { LD_LIBRARY_PATH: "" }
}
