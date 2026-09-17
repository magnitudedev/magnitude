import { FileSystem, Path } from "@effect/platform"
import { Context, Effect, Layer, Option, Schema } from "effect"

export class CliBinaryUnavailable extends Schema.TaggedError<CliBinaryUnavailable>()(
  "CliBinaryUnavailable", { reason: Schema.String },
) {}

export interface DesktopCli {
  readonly executable: string
  readonly application: string
}

export interface CliBinaryResolver {
  readonly resolve: Effect.Effect<DesktopCli, CliBinaryUnavailable>
}
export const CliBinaryResolver = Context.GenericTag<CliBinaryResolver>("launcher/CliBinaryResolver")

export interface DesktopLocation {
  readonly platform: string
  readonly home: string
  readonly application: Option.Option<string>
  readonly localAppData: Option.Option<string>
}

/** Only known desktop locations are considered; searching PATH could find this launcher itself. */
export const cliBinaryResolverLayer = (location: DesktopLocation) => Layer.effect(CliBinaryResolver, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  const unavailable = () => new CliBinaryUnavailable({
    reason: "Magnitude desktop is not installed or its bundled CLI is unavailable. Download and install Magnitude from https://magnitude.dev, then run this command again.",
  })
  const resolve = Effect.gen(function* () {
    if (!["darwin", "linux", "win32"].includes(location.platform)) return yield* unavailable()
    const applications = Option.match(location.application, {
      onSome: application => [application],
      onNone: () => location.platform === "darwin"
        ? ["/Applications/Magnitude.app", path.join(location.home, "Applications", "Magnitude.app")]
        : location.platform === "linux" ? ["/usr/bin/magnitude-desktop"]
        : Option.match(location.localAppData, {
          onNone: () => [],
          onSome: directory => [path.join(directory, "Programs", "Magnitude", "Magnitude.exe")],
        }),
    })
    for (const application of applications) {
      if (!path.isAbsolute(application)) continue
      const executable = location.platform === "darwin"
        ? path.join(application, "Contents", "Resources", "magnitude")
        : location.platform === "linux" && application === "/usr/bin/magnitude-desktop"
        ? "/usr/lib/magnitude-desktop/resources/magnitude"
        : path.join(path.dirname(application), "resources", location.platform === "win32" ? "magnitude.exe" : "magnitude")
      const appExecutable = location.platform === "darwin" ? path.join(application, "Contents", "MacOS", "Magnitude") : application
      const usable = yield* Effect.forEach([appExecutable, executable], file => fs.stat(file).pipe(
        Effect.flatMap(info => info.type === "File" && (location.platform === "win32" || (info.mode & 0o111) !== 0) ? fs.access(file).pipe(Effect.as(true)) : Effect.succeed(false)),
        Effect.catchAll(() => Effect.succeed(false)),
      ))
      if (usable.every(Boolean)) return { application, executable }
    }
    return yield* unavailable()
  })
  return { resolve }
}))
