import { Command, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Schema } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { join, resolve } from "node:path"
import { acceptanceInferenceInstallation } from "./inference-installation"
import { appleSigning, signAppleCode } from "../apple/signing"
import { compileAppleBun } from "../apple/compile-bun"
import { desktopUpdateArchive } from "../../src/targets"
import { sha256File } from "../../src/macos-app"
import { signUpdateRelease, UpdateRelease } from "../../src/hosted-update/release"
import { signUpdateManifest, UpdateManifest } from "../../src/hosted-update/manifest"
import { writeInstallationDistribution } from "../build/installation-distribution"
import { makePreparedUpdateStore } from "../../../daemon-management/src/desktop-native/prepared-update"
import { unixPrivateFilePermissions } from "../../../daemon-management/src/desktop-native/private-files"

class AcceptanceFailed extends Schema.TaggedError<AcceptanceFailed>()("AcceptanceFailed", { message: Schema.String }) {}
const Configuration = Schema.Struct({ origin: Schema.String, keyId: Schema.String, publicKey: Schema.String })
const HarnessInvocation = Schema.Struct({ resources: Schema.String, stateDirectory: Schema.String, dataDirectory: Schema.String,
  version: Schema.String, architecture: Schema.Literal("arm64"), operation: Schema.Literal("Install", "Recover"), continuation: Schema.TaggedStruct("None", {}) })
const Evidence = Schema.Struct({ installedVersion: Schema.String, preparedRecordRetired: Schema.Literal(true),
  helperRetired: Schema.Literal(true), transactionRetired: Schema.Literal(true), scriptInstallation: Schema.Literal(true) })
const root = resolve(import.meta.dir, "../../../..")
const run = Effect.gen(function* () {
  if (process.platform !== "darwin" || process.arch !== "arm64") return yield* new AcceptanceFailed({ message: "Signed fixture requires a native macOS arm64 runner" })
  if ((yield* appleSigning).mode !== "developer-id") return yield* new AcceptanceFailed({ message: "This acceptance requires the production publisher signing path" })
  const fs = yield* FileSystem.FileSystem
  const output = resolve(yield* Config.string("MAGNITUDE_HEADLESS_ACCEPTANCE_OUTPUT"))
  yield* fs.makeDirectory(output, { mode: 0o700 })
  const inference = yield* acceptanceInferenceInstallation(join(output, "inference"))
  const keys = yield* Effect.sync(() => generateKeyPairSync("ed25519"))
  const config = join(output, "configuration.json")
  yield* fs.writeFileString(config, yield* Schema.encode(Schema.parseJson(Configuration))({ origin: "http://127.0.0.1:9", keyId: "isolated", publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString() }), { mode: 0o600 })
  const command = (executable: string, args: readonly string[], environment: Record<string, string> = {}) => Command.make(executable, ...args).pipe(
    Command.workingDirectory(root), Command.env(environment), Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
    Effect.filterOrFail(code => code === 0, code => new AcceptanceFailed({ message: `Acceptance command exited ${code}: ${executable}` })), Effect.asVoid)
  const versions = ["0.0.501", "0.0.502", "0.0.503", "0.0.504"] as const
  const releases = new Map<string, UpdateRelease>()
  for (const version of versions) {
    yield* command(process.execPath, [join(import.meta.dir, "build-desktop.ts")], {
      MAGNITUDE_ACCEPTANCE_VERSION: version, MAGNITUDE_ACCEPTANCE_OUTPUT: join(output, version),
      MAGNITUDE_ACCEPTANCE_CONFIG: config, MAGNITUDE_ACCEPTANCE_STANDARD_BUNDLE_ID: "true",
    })
    const archive = join(output, version, "artifacts", desktopUpdateArchive("darwin-arm64"))
    const release = yield* signUpdateRelease({ version, bytes: Number((yield* fs.stat(archive)).size), sha256: yield* sha256File(archive) },
      { os: "darwin", arch: "arm64", package: "mac-zip" }, keys.privateKey)
    releases.set(version, release)
    yield* fs.writeFileString(join(output, version, "release.json"), yield* Schema.encode(Schema.parseJson(UpdateRelease))(release))
  }
  const installed = join(output, "installed")
  yield* fs.makeDirectory(installed, { mode: 0o700 })
  const bundle = join(installed, "Magnitude.app"), resources = join(bundle, "Contents/Resources")
  const stateDirectory = join(output, "state"), dataDirectory = join(output, "profile")
  const environment = { MAGNITUDE_DEV_DATA_DIR: dataDirectory, MAGNITUDE_DESKTOP_STATE_DIR: stateDirectory, MAGNITUDE_DEV_PORT: "11237", MAGNITUDE_ICN_PATH: inference }
  const archiveName = desktopUpdateArchive("darwin-arm64")
  const initialArchive = join(output, versions[0], "artifacts", archiveName)
  const manifest = yield* Schema.decodeUnknown(UpdateManifest)({ protocol: 1, version: versions[0], tag: `@magnitudedev/cli@${versions[0]}`, commit: "a".repeat(40),
    artifact: { id: "desktop-update-darwin-arm64", target: { os: "darwin", arch: "arm64", package: "mac-zip" }, filename: archiveName,
      bytes: Number((yield* fs.stat(initialArchive)).size), sha256: yield* sha256File(initialArchive) } })
  const hosting = join(output, "script-hosting")
  yield* writeInstallationDistribution({ output: hosting, origin: "https://localhost:18443", appleTeam: yield* Config.string("APPLE_TEAM_ID"),
    windowsPublisher: "Magnitude Update Acceptance", publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString(),
    publications: [yield* signUpdateManifest(manifest, keys.privateKey)] })
  const downloadDirectory = join(hosting, "magnitudedev/magnitude/releases/download", manifest.tag)
  yield* fs.makeDirectory(downloadDirectory, { recursive: true })
  yield* fs.copyFile(initialArchive, join(downloadDirectory, archiveName))
  yield* command("/bin/bash", [join(import.meta.dir, "test-mac-install-script.sh"), hosting, bundle, versions[0], dataDirectory, stateDirectory], environment)
  yield* command(process.execPath, [join(import.meta.dir, "test-installed-headless.ts")], {
    MAGNITUDE_INSTALLED_ACCEPTANCE_OUTPUT: join(output, "script-serve"),
    MAGNITUDE_INSTALLED_ACCEPTANCE_CLI: join(resources, "magnitude"),
    MAGNITUDE_INSTALLED_ACCEPTANCE_ADDON: join(resources, "desktop-host.node"),
    MAGNITUDE_INSTALLED_ACCEPTANCE_VERSION: versions[0], MAGNITUDE_ICN_PATH: inference,
  })
  const harness = join(output, "installer-entry")
  yield* compileAppleBun(join(import.meta.dir, "mac-foreground-installer-entry.ts"), harness, "bun-darwin-arm64", "cli")
  yield* signAppleCode(harness, "dev.magnitude.installer-acceptance", "bun")
  for (const [previous, replacement] of [[versions[0], versions[1]], [versions[1], versions[2]], [versions[2], versions[3]]] as const) {
    const archive = join(output, replacement, "artifacts", desktopUpdateArchive("darwin-arm64"))
    const release = releases.get(replacement)!
    const transfer = join(output, "transfer.zip")
    yield* fs.copyFile(archive, transfer)
    yield* Effect.gen(function* () {
      const store = yield* makePreparedUpdateStore({ dataDirectory, target: { os: "darwin", arch: "arm64", package: "mac-zip" }, trustedPublishers: new Map([["isolated", keys.publicKey]]) })
      yield* store.prepare(transfer, release)
    }).pipe(Effect.provide(unixPrivateFilePermissions))
    if (previous === versions[0]) yield* command(join(resources, "magnitude"), ["update", "install"], environment)
    else yield* Effect.scoped(Effect.gen(function* () {
      const desktop = previous === versions[2]
      const invocation = desktop ? Command.make(join(bundle, "Contents/MacOS/Magnitude"), "--background") : Command.make(join(resources, "magnitude"), "serve")
      const serving = yield* invocation.pipe(Command.env(environment),
        Command.stdout("inherit"), Command.stderr("inherit"), Command.start)
      yield* Effect.gen(function* () {
        for (;;) {
          if (!(yield* serving.isRunning)) return yield* new AcceptanceFailed({ message: "Foreground startup exited before the replacement became ready" })
          const status = yield* Command.make(join(resources, "magnitude"), "status").pipe(Command.env(environment), Command.string)
          if (/Runtime\s+Ready/.test(status) && (desktop ? /Owner\s+Desktop/ : /Owner\s+Headless/).test(status) && status.includes(replacement)) break
          yield* Effect.sleep("500 millis")
        }
      }).pipe(Effect.timeout("5 minutes"))
      yield* serving.kill("SIGTERM")
      const exit = yield* serving.exitCode.pipe(Effect.timeout("30 seconds"))
      if (exit !== 0) return yield* new AcceptanceFailed({ message: `Foreground replacement shutdown exited ${exit}` })
    }))
    yield* command(harness, [yield* Schema.encode(Schema.parseJson(HarnessInvocation))({ resources, stateDirectory, dataDirectory, version: replacement, architecture: "arm64", operation: "Recover", continuation: { _tag: "None" } })])
    const actual = (yield* Command.make(join(resources, "magnitude"), "--version").pipe(Command.string)).trim()
    const serviceVersion = (yield* Command.make(join(resources, "magnitude-service"), "--version").pipe(Command.string)).trim()
    if (actual !== replacement || serviceVersion !== replacement || (yield* fs.exists(join(dataDirectory, "updates/update.json"))) ||
        (yield* fs.exists(join(installed, ".Magnitude.app.update"))) || (yield* fs.readDirectory(join(stateDirectory, "mac-installers"))).length) {
      return yield* new AcceptanceFailed({ message: "Installed version or transaction retirement did not match the signed update" })
    }
    yield* command("/usr/bin/codesign", ["--verify", "--deep", "--strict", bundle])
    yield* command("/usr/bin/xcrun", ["stapler", "validate", bundle])
    yield* command("/usr/sbin/spctl", ["--assess", "--type", "execute", "--verbose", bundle])
  }
  yield* fs.writeFileString(join(output, "result.json"), yield* Schema.encode(Schema.parseJson(Evidence))({ installedVersion: versions[3], preparedRecordRetired: true, helperRetired: true, transactionRetired: true, scriptInstallation: true }))
  yield* Effect.logInfo("Signed macOS finite, foreground and desktop installer acceptance passed")
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
