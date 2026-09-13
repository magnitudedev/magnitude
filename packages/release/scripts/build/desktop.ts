import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Schema } from "effect"
import { packager } from "@electron/packager"
import { resolve, join, dirname } from "node:path"
import { fileURLToPath } from "node:url"
import { ACN_EXECUTABLE_NAME } from "../../src/executables"
import { MACOS_DEPLOYMENT_TARGET } from "../../src/targets"

export class DesktopBuildFailed extends Schema.TaggedError<DesktopBuildFailed>()("DesktopBuildFailed", { message: Schema.String }) {}
const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..")
const PackageVersion = Schema.Struct({ version: Schema.NonEmptyString })
const ApplicationPackage = Schema.Struct({ name: Schema.String, version: Schema.String, type: Schema.Literal("module"), main: Schema.String })
export const DesktopTarget = Schema.Union(
  Schema.Struct({ platform: Schema.Literal("darwin", "linux"), arch: Schema.Literal("arm64", "x64") }),
  Schema.Struct({ platform: Schema.Literal("win32"), arch: Schema.Literal("x64") }),
)

/** Assemble the desktop and its exact service together. Signing/notarization follow assembly. */
export const buildDesktopApplication = (options: {
  readonly service: string
  readonly cli: string
  readonly outputDirectory: string
  readonly version: string
  readonly revision: number
  readonly target?: typeof DesktopTarget.Type
}) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const { platform, arch } = yield* Schema.decodeUnknown(DesktopTarget)(options.target ?? { platform: process.platform, arch: process.arch })
  const stage = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-desktop-package-" })
  const app = join(stage, "app")
  const resources = join(stage, "resources")
  yield* fs.makeDirectory(app)
  yield* fs.makeDirectory(resources)
  yield* fs.copy(join(root, "desktop/out"), join(app, "out"))
  yield* fs.writeFileString(join(app, "package.json"), yield* Schema.encode(Schema.parseJson(ApplicationPackage))({
    name: "magnitude-desktop", version: options.version, type: "module", main: "out/main/main.js",
  }))
  const serviceName = `${ACN_EXECUTABLE_NAME}${platform === "win32" ? ".exe" : ""}`
  const service = join(resources, serviceName)
  const cli = join(resources, platform === "win32" ? "magnitude.exe" : "magnitude")
  const addon = join(resources, "desktop-host.node")
  const tray = join(resources, "trayTemplate@2x.png")
  const icon = join(resources, "application-icon.png")
  const license = join(resources, "Magnitude-LICENSE.txt")
  const command = join(resources, "magnitude-command")
  yield* fs.copyFile(options.service, service)
  yield* fs.chmod(service, 0o755)
  yield* fs.copyFile(options.cli, cli)
  yield* fs.chmod(cli, 0o755)
  yield* fs.copyFile(join(root, `packages/daemon-management/dist/native/${platform}-${arch}/desktop-host.node`), addon)
  yield* fs.copyFile(join(root, "assets/brand/trayTemplate@2x.png"), tray)
  yield* fs.copyFile(join(root, "assets/brand/application-icon.png"), icon)
  yield* fs.copyFile(join(root, "LICENSE"), license)
  if (platform !== "win32") {
    yield* fs.copyFile(join(root, `packages/daemon-management/dist/native/${platform}-${arch}/magnitude-command`), command)
    yield* fs.chmod(command, 0o755)
  }
  const electron = yield* fs.readFileString(join(root, "node_modules/electron/package.json")).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(PackageVersion))))
  return yield* Effect.tryPromise({ try: () => packager({
    dir: app, out: options.outputDirectory, name: "Magnitude", executableName: platform === "linux" ? "magnitude" : "Magnitude",
    platform, arch, electronVersion: electron.version,
    appBundleId: "dev.magnitude.desktop", appVersion: options.version, buildVersion: String(options.revision),
    ...(platform === "win32" ? { icon: join(root, "packages/release/resources/windows/Magnitude.ico"), win32metadata: { CompanyName: "Magnitude" } } : {}),
    ...(platform === "darwin" ? { icon: join(root, "packages/release/resources/macos/Magnitude.icns"), extendInfo: { LSMinimumSystemVersion: MACOS_DEPLOYMENT_TARGET } } : {}),
    asar: true, prune: false, overwrite: true,
    extraResource: [service, cli, addon, tray, icon, license, ...(platform === "win32" ? [] : [command])],
  }), catch: error => new DesktopBuildFailed({ message: `Could not assemble desktop: ${String(error)}` }) })
})).pipe(Effect.mapError(error => error instanceof DesktopBuildFailed ? error : new DesktopBuildFailed({ message: String(error) })))
