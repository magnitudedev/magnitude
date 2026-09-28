import * as Command from "@effect/platform/Command"
import { Config, Data, Effect, Schema } from "effect"
import { fileURLToPath } from "node:url"
import { WindowsPublisher } from "../../src/desktop-distribution"

export class WindowsSigningFailed extends Data.TaggedError("WindowsSigningFailed")<{
  readonly message: string
}> {}

/** Artifact Signing requires the publisher identity that signing verifies and Desktop trusts. */
export const windowsSigning = Effect.gen(function* () {
  const mode = yield* Config.literal("unsigned", "artifact-signing")("MAGNITUDE_WINDOWS_DISTRIBUTION").pipe(
    Config.withDefault("unsigned"),
  )
  if (mode === "unsigned") return { mode } as const
  const publisher = yield* Schema.decodeUnknown(WindowsPublisher)(yield* Config.string("MAGNITUDE_WINDOWS_PUBLISHER")).pipe(
    Effect.mapError(() => new WindowsSigningFailed({ message: "Artifact Signing requires a Windows publisher identity" })),
  )
  return { mode, publisher } as const
})

export const windowsSigningScript = fileURLToPath(new URL("./windows-signing.ps1", import.meta.url))


/** Verify publisher and timestamp before any signed bytes enter an archive or installer. */
export const signWindowsCode = (file: string) => Effect.gen(function* () {
  if ((yield* windowsSigning).mode === "unsigned") return
  const code = yield* Command.make("pwsh.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", windowsSigningScript, "-Path", file).pipe(
    Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
  )
  if (code !== 0) return yield* new WindowsSigningFailed({ message: `Windows signing failed for ${file} (exit ${code})` })
})
