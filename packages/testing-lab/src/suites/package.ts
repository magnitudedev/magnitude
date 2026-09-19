import { Effect, Schema } from "effect"
import { join } from "node:path"
import { ACN_EXECUTABLE_NAME } from "@magnitudedev/release/executables"
import { AssertionFailure } from "../domain"
import { InstalledApplication } from "../installer"
import { inspectNativeImage, NativeImage } from "../native-image"
import { command } from "../process"

export const PackageIdentity = Schema.Struct({ version: Schema.String, binaries: Schema.Array(Schema.Struct({ name: Schema.String, image: NativeImage })) })
/** Inspect the installed payload and execute its own service/CLI; ambient binaries cannot satisfy this check. */
export const inspectPackageIdentity = (app: InstalledApplication, observedDesktopVersion: string, environment: Readonly<Record<string, string>>) => Effect.gen(function* () {
  const target = app.candidate.target, expected = app.candidate.version
  if (observedDesktopVersion !== expected) return yield* new AssertionFailure({ message: "Running desktop version differs from admitted package" })
  const resources = join(app.root, target.os === "macos" ? "Contents/Resources" : "resources")
  const suffix = target.os === "windows" ? ".exe" : ""
  const paths = [
    { name: "desktop", path: target.os === "macos" || target.os === "windows" ? app.executable : join(app.root, "magnitude") },
    { name: "service", path: join(resources, `${ACN_EXECUTABLE_NAME}${suffix}`) },
    { name: "cli", path: join(resources, `magnitude${suffix}`) },
    { name: "native-host", path: join(resources, "desktop-host.node") },
    ...(target.os === "windows" ? [] : [{ name: "command-helper", path: join(resources, "magnitude-command") }]),
  ]
  const format = target.os === "macos" ? "mach-o" : target.os === "windows" ? "pe" : "elf"
  const binaries = yield* Effect.forEach(paths, file => Effect.gen(function* () {
    const image = yield* inspectNativeImage(file.path)
    if (image.format !== format || !image.architectures.includes(target.arch)) return yield* new AssertionFailure({ message: `${file.name} does not contain the requested native OS/CPU architecture` })
    if (file.name === "service" || file.name === "cli") {
      const output = yield* command(file.path, [file.name === "cli" ? "--version" : "version"], { env: { ...environment }, inheritEnv: false, timeoutMs: 30_000 })
      if (output.exitCode !== 0 || output.stdout.trim() !== expected) return yield* new AssertionFailure({ message: `Bundled ${file.name} version differs from the running desktop and admitted package` })
    }
    return { name: file.name, image }
  }))
  return PackageIdentity.make({ version: expected, binaries })
})
