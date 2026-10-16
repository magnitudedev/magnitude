import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Schema } from "effect"
import { packager, type Options as PackagerOptions } from "@electron/packager"
import { resolve, join, dirname, basename } from "node:path"
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

/**
 * Electron packaging for one target. macOS registers `magnitude://` links in the bundle's Info.plist.
 * Windows shows `buildVersion` as the executable's file version; macOS keeps the revision as its bundle version.
 */
export const desktopPackagerOptions = (options: {
  readonly app: string
  readonly outputDirectory: string
  readonly target: typeof DesktopTarget.Type
  readonly electronVersion: string
  readonly version: string
  readonly revision: number
  readonly extraResource: ReadonlyArray<string>
}): PackagerOptions => {
  const { platform, arch } = options.target
  return {
    dir: options.app, out: options.outputDirectory, name: "Magnitude", executableName: platform === "linux" ? "magnitude" : "Magnitude",
    platform, arch, electronVersion: options.electronVersion,
    appBundleId: "dev.magnitude.desktop", appVersion: options.version, appCopyright: "Copyright Magnitude",
    buildVersion: platform === "win32" ? `${options.version.split(/[+-]/)[0]}.${options.revision}` : String(options.revision),
    ...(platform === "win32" ? { icon: join(root, "packages/release/resources/windows/Magnitude.ico"),
      win32metadata: { CompanyName: "Magnitude", ProductName: "Magnitude", FileDescription: "Magnitude", InternalName: "Magnitude", OriginalFilename: "Magnitude.exe" } } : {}),
    ...(platform === "darwin" ? { icon: join(root, "packages/release/resources/macos/Magnitude.icns"), extendInfo: { LSMinimumSystemVersion: MACOS_DEPLOYMENT_TARGET },
      protocols: [{ name: "Magnitude", schemes: ["magnitude"] }] } : {}),
    asar: true, prune: false, overwrite: true,
    extraResource: [...options.extraResource],
  }
}

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
  const target = yield* Schema.decodeUnknown(DesktopTarget)(options.target ?? { platform: process.platform, arch: process.arch })
  const { platform, arch } = target
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
  const extractor = join(resources, "magnitude-extract")
  const trayStatus = join(resources, "tray-status.node")
  const launcher = join(resources, "magnitude-launcher.exe")
  const updateConfiguration = join(resources, "update-configuration.json")
  yield* fs.copyFile(join(root, "desktop/out/main/update-configuration.json"), updateConfiguration)
  const updateTrust = join(resources, "update-trust.json")
  yield* fs.copyFile(join(root, "desktop/out/main/update-trust.json"), updateTrust)
  yield* fs.copyFile(options.service, service)
  yield* fs.chmod(service, 0o755)
  yield* fs.copyFile(options.cli, cli)
  yield* fs.chmod(cli, 0o755)
  yield* fs.copyFile(join(root, `packages/daemon-management/dist/native/${platform}-${arch}/desktop-host.node`), addon)
  yield* fs.copyFile(join(root, "assets/brand/trayTemplate@2x.png"), tray)
  const windowsTrayIcons = platform === "win32" ? ["tray-black.ico", "tray-white.ico"].map(name => join(resources, name)) : []
  for (const path of windowsTrayIcons) yield* fs.copyFile(join(root, "assets/brand", basename(path)), path)
  yield* fs.copyFile(join(root, "assets/brand/application-icon.png"), icon)
  yield* fs.copyFile(join(root, "LICENSE"), license)
  if (platform === "win32") yield* fs.copyFile(join(root, "packages/daemon-management/dist/native/win32-x64/magnitude-launcher.exe"), launcher)
  if (platform !== "win32") {
    yield* fs.copyFile(join(root, `packages/daemon-management/dist/native/${platform}-${arch}/magnitude-command`), command)
    yield* fs.chmod(command, 0o755)
  }
  if (platform === "darwin") {
    yield* fs.copyFile(join(root, `packages/daemon-management/dist/native/${platform}-${arch}/magnitude-extract`), extractor)
    yield* fs.chmod(extractor, 0o755)
    yield* fs.copyFile(join(root, `desktop/dist/native/${platform}-${arch}/tray-status.node`), trayStatus)
  }
  const electron = yield* fs.readFileString(join(root, "node_modules/electron/package.json")).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(PackageVersion))))
  return yield* Effect.tryPromise({ try: () => packager(desktopPackagerOptions({
    app, outputDirectory: options.outputDirectory, target, electronVersion: electron.version,
    version: options.version, revision: options.revision,
    extraResource: [service, cli, addon, tray, ...windowsTrayIcons, icon, license, updateTrust, updateConfiguration,
      ...(platform === "win32" ? [launcher] : [command]), ...(platform === "darwin" ? [extractor, trayStatus] : [])],
  })), catch: error => new DesktopBuildFailed({ message: `Could not assemble desktop: ${String(error)}` }) })
})).pipe(Effect.mapError(error => error instanceof DesktopBuildFailed ? error : new DesktopBuildFailed({ message: String(error) })))
