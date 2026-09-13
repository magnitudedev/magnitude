import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { Option } from "effect"
import { makeDesktopApplicationHost } from "@magnitudedev/daemon-management/desktop-native"
import { bundledWindowsNative } from "@magnitudedev/daemon-management/bun"
import { isDevelopmentBuild } from "../runtime/environment"

export const {
  desktopIsolatedProfile, desktopDataDirectory, desktopServiceOrigin, desktopApplication,
  startDesktopApplication, stopDesktopApplication,
  readDesktopLoginStartup, setDesktopLoginStartup, updateDesktopApplication,
} = makeDesktopApplicationHost(isDevelopmentBuild()
  ? Option.some(resolve(dirname(fileURLToPath(import.meta.url)), "../../.."))
  : Option.none(), bundledWindowsNative)
