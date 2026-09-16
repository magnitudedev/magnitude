import { access, realpath } from "node:fs/promises"
import { basename, dirname, join } from "node:path"
import { Effect } from "effect"
import { ApplicationLaunchFailed } from "./application-client"

/** A bundled CLI follows its own app, including when invoked through /usr/local/bin. */
export const resolveMacApplicationPath = (executable: string, home: string, override?: string) =>
  Effect.tryPromise({
    try: async () => {
      if (override) return override
      const resolved = await realpath(executable)
      const resources = dirname(resolved)
      const contents = dirname(resources)
      const bundle = dirname(contents)
      if (basename(resolved) === "magnitude" && basename(resources) === "Resources" &&
          basename(contents) === "Contents" && bundle.endsWith(".app")) return bundle
      // Standalone diagnostic builds retain the standard desktop discovery locations.
      for (const candidate of ["/Applications/Magnitude.app", join(home, "Applications/Magnitude.app")]) {
        try { await access(join(candidate, "Contents/Frameworks/Electron Framework.framework")); return candidate }
        catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error }
      }
      throw new Error("Install the Magnitude desktop app from magnitude.dev before starting inference.")
    },
    catch: error => new ApplicationLaunchFailed({ message: `Could not locate Magnitude: ${String(error)}` }),
  })
