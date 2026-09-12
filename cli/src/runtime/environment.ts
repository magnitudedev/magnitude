import { isDevelopmentVersion } from "../update/updater"
import { CLI_VERSION } from "../version"

export const isDevelopmentBuild = (): boolean =>
  /\.tsx?$/.test(import.meta.url)
  || (process.argv[1]?.endsWith(".tsx") ?? false)
  || isDevelopmentVersion(CLI_VERSION)
