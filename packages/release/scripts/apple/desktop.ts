import * as FileSystem from "@effect/platform/FileSystem"
import { Config, Effect, Option, Schema } from "effect"
import { sign, walk } from "@electron/osx-sign"
import { basename, join, resolve } from "node:path"
import { ReleaseArtifactSchema } from "../../src/contracts"
import { sha256File } from "../../src/macos-app"
import { ACN_EXECUTABLE_NAME } from "../../src/executables"
import { appleRequirement } from "../../src/trust"
import { desktopInstaller, desktopUpdateArchive } from "../../src/targets"
import * as Command from "@effect/platform/Command"
import { AppleDistributionFailed, appleCommand, appleSigning } from "./signing"
import { notarizeAppleUnit } from "./distribution"
import { verifyAppleDeploymentTarget } from "../build/common"

const resources = resolve(import.meta.dir, "../../resources/macos")

export const validateDesktopDistribution = (options: { readonly image: string; readonly updateArchive: string; readonly version: string; readonly revision: number; readonly rpcVersion: number; readonly inferenceInstallation: string }) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const signing = yield* appleSigning
  const mount = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-dmg-consumer-" })
  yield* Effect.acquireRelease(
    appleCommand("/usr/bin/hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", mount, options.image),
    () => appleCommand("/usr/bin/hdiutil", "detach", mount).pipe(Effect.orDie),
  )
  const extracted = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-update-consumer-" })
  yield* appleCommand("/usr/bin/ditto", "-x", "-k", options.updateArchive, extracted)
  // Both distributions must carry the same signed app, including framework symlinks.
  yield* appleCommand("/usr/bin/diff", "-qr", join(mount, "Magnitude.app"), join(extracted, "Magnitude.app"))
  for (const app of [join(mount, "Magnitude.app"), join(extracted, "Magnitude.app")]) {
    yield* appleCommand("/usr/bin/codesign", "--verify", "--deep", "--strict", "-R", `=${appleRequirement("dev.magnitude.desktop", signing.team)}`, app)
    if (signing.mode === "developer-id") yield* appleCommand("/usr/bin/xcrun", "stapler", "validate", app)
    const version = yield* appleCommand(join(app, "Contents/Resources", ACN_EXECUTABLE_NAME), "version")
    if (version.trim() !== options.version) return yield* new AppleDistributionFailed({ message: "Desktop contains a different service release" })
    const code = yield* Command.make(process.env.MAGNITUDE_TEST_NODE ?? "node", resolve(import.meta.dir, "../../../../desktop/src/fixtures/packaged-lifecycle.mjs")).pipe(
      Command.env({ MAGNITUDE_TEST_DESKTOP_EXECUTABLE: join(app, "Contents/MacOS/Magnitude"), MAGNITUDE_ICN_PATH: options.inferenceInstallation,
        MAGNITUDE_TEST_EXPECT_VERSION: options.version, MAGNITUDE_TEST_EXPECT_REVISION: String(options.revision), MAGNITUDE_TEST_EXPECT_RPC_VERSION: String(options.rpcVersion),
      }),
      Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
      Effect.timeout("3 minutes"),
    )
    if (code !== 0) return yield* new AppleDistributionFailed({ message: `Desktop lifecycle acceptance exited ${code}` })
  }
}))

/** Archive the already signed/stapled app; never sign again after installer creation. */
export const buildDesktopUpdateArchive = (options: {
  readonly app: string
  readonly output: string
  readonly host: "darwin-arm64" | "darwin-x64"
}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(options.output, { recursive: true })
  const filename = desktopUpdateArchive(options.host)
  const output = join(options.output, filename)
  yield* appleCommand("/usr/bin/ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", options.app, output)
  const artifact = yield* Schema.decodeUnknown(ReleaseArtifactSchema)({
    id: `desktop-update-${options.host}`, kind: "desktop", host: options.host,
    filename, bytes: Number((yield* fs.stat(output)).size), sha256: yield* sha256File(output),
  })
  yield* fs.writeFileString(join(options.output, `${artifact.id}.artifact.json`),
    yield* Schema.encode(Schema.parseJson(ReleaseArtifactSchema))(artifact), { flag: "wx", mode: 0o600 })
  return artifact
})

/** Preserve Electron's nested framework layout; its service needs Bun's distinct JIT profile. */
export const signDesktopApplication = (app: string) => Effect.gen(function* () {
  const signing = yield* appleSigning
  const keychain = yield* Config.string("APPLE_RELEASE_KEYCHAIN").pipe(Config.withDefault(""))
  yield* Effect.tryPromise({
    try: () => sign({
      app, platform: "darwin", identity: signing.identity,
      identityValidation: signing.mode === "developer-id",
      ...(keychain ? { keychain } : {}),
      preAutoEntitlements: false, preEmbedProvisioningProfile: false,
      optionsForFile: file => ({
        hardenedRuntime: signing.mode === "developer-id",
        ...(signing.mode === "adhoc" ? { timestamp: "none" } : {}),
        entitlements: join(resources, file.endsWith(`/${ACN_EXECUTABLE_NAME}`)
          ? "bun.entitlements.plist"
          : (/\.(node|dylib|framework)$/.test(file) || file.endsWith("/magnitude-command")) ? "library.entitlements.plist" : "electron.entitlements.plist"),
      }),
    }),
    catch: error => new AppleDistributionFailed({ message: `Could not sign desktop application: ${String(error)}` }),
  })
  yield* appleCommand("/usr/bin/codesign", "--verify", "--deep", "--strict", app)
})

/** A DMG is an explicit desktop installation, never an engine-acquisition archive. */
export const buildDesktopDmg = (options: {
  readonly app: string
  readonly output: string
  readonly host: "darwin-arm64" | "darwin-x64"
}) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const signing = yield* appleSigning
  const nativePaths = yield* Effect.tryPromise({ try: () => walk(options.app), catch: error => new AppleDistributionFailed({ message: `Could not inspect desktop native files: ${String(error)}` }) })
  // The signing walk also returns non-Mach-O binary resources, including icons.
  const nativeFiles = yield* Effect.filter(nativePaths, file => Effect.gen(function* () {
    if ((yield* fs.stat(file)).type !== "File") return false
    return (yield* appleCommand("/usr/bin/file", "-b", file)).includes("Mach-O")
  }))
  yield* Effect.tryPromise({ try: () => verifyAppleDeploymentTarget(options.host, nativeFiles), catch: error => new AppleDistributionFailed({ message: String(error) }) })
  yield* signDesktopApplication(options.app)
  const notarization = yield* notarizeAppleUnit("desktop", options.output, [options.app])
  if (signing.mode === "developer-id") {
    yield* appleCommand("/usr/bin/xcrun", "stapler", "staple", options.app)
    yield* appleCommand("/usr/bin/xcrun", "stapler", "validate", options.app)
  }
  const stage = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-dmg-" })
  yield* appleCommand("/usr/bin/ditto", options.app, join(stage, "Magnitude.app"))
  yield* fs.symlink("/Applications", join(stage, "Applications"))
  yield* fs.makeDirectory(options.output, { recursive: true })
  const filename = desktopInstaller(options.host)
  const output = join(options.output, filename)
  yield* appleCommand("/usr/bin/hdiutil", "create", "-ov", "-volname", "Magnitude", "-srcfolder", stage, "-format", "UDZO", output)
  yield* appleCommand("/usr/bin/hdiutil", "verify", output)
  const info = yield* fs.stat(output)
  const artifact = yield* Schema.decodeUnknown(ReleaseArtifactSchema)({
    id: `desktop-${options.host}`, kind: "desktop", host: options.host,
    filename: basename(output), bytes: Number(info.size), sha256: yield* sha256File(output),
  })
  yield* fs.writeFileString(join(options.output, `desktop-${options.host}.artifact.json`),
    yield* Schema.encode(Schema.parseJson(ReleaseArtifactSchema))(artifact), { flag: "wx", mode: 0o600 })
  const updateArtifact = yield* buildDesktopUpdateArchive(options)
  return { artifact, updateArtifact, notarization, stapled: Option.isSome(notarization) }
}))
