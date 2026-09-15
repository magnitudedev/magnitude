import * as Command from "@effect/platform/Command"
import { Config, Data, Effect } from "effect"
import { fileURLToPath } from "node:url"

export class WindowsSigningFailed extends Data.TaggedError("WindowsSigningFailed")<{
  readonly message: string
}> {}

export const windowsSigning = Config.literal("unsigned", "artifact-signing")("MAGNITUDE_WINDOWS_DISTRIBUTION").pipe(
  Config.withDefault("unsigned"),
)

export const windowsSigningScript = fileURLToPath(new URL("./windows-signing.ps1", import.meta.url))

/** Verify publisher and timestamp before any signed bytes enter an archive or installer. */
export const signWindowsCode = (file: string) => Effect.gen(function* () {
  if ((yield* windowsSigning) === "unsigned") return
  const code = yield* Command.make("pwsh.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", windowsSigningScript, "-Path", file).pipe(
    Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
  )
  if (code !== 0) return yield* new WindowsSigningFailed({ message: `Windows signing failed for ${file} (exit ${code})` })
})
