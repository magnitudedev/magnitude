import { CLI_VERSION } from "../version"

const isDevelopmentVersion = (version: string): boolean => version.includes("+dev.") || version === "0.0.0"

export const isDevelopmentBuild = (): boolean =>
  /\.tsx?$/.test(import.meta.url)
  || (process.argv[1]?.endsWith(".tsx") ?? false)
  || isDevelopmentVersion(CLI_VERSION)
