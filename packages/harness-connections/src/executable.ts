import * as Command from "@effect/platform/Command"
import { Effect, Schema } from "effect"
import { accessSync, constants, statSync } from "node:fs"
import { delimiter, extname, resolve } from "node:path"

export class HarnessCommandUnsupported extends Schema.TaggedError<HarnessCommandUnsupported>()("HarnessCommandUnsupported", {
  message: Schema.String,
}) {}

const DEFAULT_PATHEXT = ".COM;.EXE;.BAT;.CMD"

/**
 * Resolve an installed executable without invoking a shell or starting the harness.
 * Windows only launches files with a PATHEXT extension, so an extensionless sibling
 * such as npm's POSIX shim is never a Windows installation.
 */
export const findExecutable = (
  name: string,
  searchPath: string,
  platform: NodeJS.Platform = process.platform,
  pathext: string | undefined = process.env.PATHEXT,
): string | undefined => {
  const windows = platform === "win32"
  const suffixes = windows
    ? [...(extname(name) === "" ? [] : [""]), ...(pathext ?? DEFAULT_PATHEXT).split(";").filter(Boolean)]
    : [""]
  for (const directory of searchPath.split(windows ? ";" : delimiter).filter(Boolean)) {
    for (const suffix of suffixes) {
      const candidate = resolve(directory, `${name}${suffix}`)
      try {
        accessSync(candidate, windows ? constants.F_OK : constants.X_OK)
        if (statSync(candidate).isFile()) return candidate
      } catch { /* Missing or inaccessible executable is not an installation. */ }
    }
  }
  return undefined
}

const isWindowsBatch = (executable: string) => /\.(?:cmd|bat)$/i.test(executable)

/**
 * Build the command that runs a detected harness executable. Windows batch files
 * (such as npm's `.cmd` shims) can only run through `cmd.exe`, which receives one
 * quoted command line; tokens cmd.exe would reinterpret inside quotes are rejected.
 */
export const harnessCommand = (
  executable: string,
  args: ReadonlyArray<string>,
  platform: NodeJS.Platform = process.platform,
): Effect.Effect<Command.Command, HarnessCommandUnsupported> => {
  if (platform !== "win32" || !isWindowsBatch(executable)) return Effect.succeed(Command.make(executable, ...args))
  const tokens = [executable, ...args]
  const unsafe = tokens.find((token) => /["%\r\n]/.test(token))
  if (unsafe !== undefined) {
    return Effect.fail(new HarnessCommandUnsupported({ message: `Cannot run ${executable} through cmd.exe with the argument ${JSON.stringify(unsafe)}.` }))
  }
  const line = tokens.map((token) => /^[\w.:\\/=@+-]+$/.test(token) ? token : `"${token}"`).join(" ")
  return Effect.succeed(Command.make(line).pipe(Command.runInShell(true)))
}
