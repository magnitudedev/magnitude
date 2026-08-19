import { isDevelopmentVersion } from "../features/update/updater"
import { CLI_VERSION } from "../version"

export const isDevelopmentBuild = (): boolean =>
  import.meta.url.endsWith(".tsx")
  || (process.argv[1]?.endsWith(".tsx") ?? false)
  || isDevelopmentVersion(CLI_VERSION)
