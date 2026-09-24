import { Command, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Option, Schema } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { join, resolve } from "node:path"
import { acceptanceInferenceInstallation } from "./inference-installation"
import { windowsDesktopInstaller } from "../../src/targets"
import { sha256File } from "../../src/macos-app"
import { signUpdateRelease, UpdateRelease } from "../../src/hosted-update/release"
import { makePreparedUpdateStore } from "../../../daemon-management/src/desktop-native/prepared-update"
import { PrivateFilePermissions, windowsPrivateFilePermissions } from "../../../daemon-management/src/desktop-native/private-files"
import { NativeHost, nativeHostLayer } from "../../../daemon-management/src/desktop-native/index"
import { requestApplication } from "../../../daemon-management/src/desktop-native/application-control"

class AcceptanceFailed extends Schema.TaggedError<AcceptanceFailed>()("AcceptanceFailed", { message: Schema.String }) {}
const Configuration = Schema.Struct({ origin: Schema.String, keyId: Schema.String, publicKey: Schema.String, windowsPublisher: Schema.String })
const Evidence = Schema.Struct({ installedVersion: Schema.String, finiteInstall: Schema.Literal(true), foregroundContinuation: Schema.Literal(true), gracefulExit: Schema.Literal(true), desktopCliRegression: Schema.Literal(true) })
const root = resolve(import.meta.dir, "../../../..")
const run = Effect.gen(function* () {
  if (process.platform !== "win32" || process.arch !== "x64") return yield* new AcceptanceFailed({ message: "Requires the Windows x64 product runtime" })
  const fs = yield* FileSystem.FileSystem
  const output = resolve(yield* Config.string("MAGNITUDE_HEADLESS_ACCEPTANCE_OUTPUT"))
  yield* fs.makeDirectory(output, { recursive: true })
  const nativeAddon = join(output, "acceptance-host.node")
  yield* fs.copyFile(join(root, "packages/daemon-management/dist/native/win32-x64/desktop-host.node"), nativeAddon)
  const native = yield* NativeHost.pipe(Effect.provide(nativeHostLayer(nativeAddon)))
  const local = yield* native.localAppDataDirectory
  const application = join(local, "Programs/Magnitude"), resources = join(application, "resources")
  if (yield* fs.exists(application)) return yield* new AcceptanceFailed({ message: "Requires an unused disposable application installation" })
  const launcher = join(local, "Programs/Magnitude CLI/magnitude.exe")
  const inference = yield* acceptanceInferenceInstallation(join(output, "inference"))
  const keys = yield* Effect.sync(() => generateKeyPairSync("ed25519"))
  const config = join(output, "configuration.json")
  yield* fs.writeFileString(config, yield* Schema.encode(Schema.parseJson(Configuration))({ origin: "http://127.0.0.1:9", keyId: "isolated",
    publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString(), windowsPublisher: "Magnitude Update Acceptance" }))
  const dataDirectory = join(output, "profile"), stateDirectory = join(output, "state")
  yield* Effect.flatMap(PrivateFilePermissions, permissions => permissions.prepareDirectory(dataDirectory)).pipe(
    Effect.provide(windowsPrivateFilePermissions(nativeAddon)))
  const environment = { MAGNITUDE_DEV_DATA_DIR: dataDirectory, MAGNITUDE_DESKTOP_STATE_DIR: stateDirectory,
    MAGNITUDE_DEV_PORT: "11237", MAGNITUDE_ICN_PATH: inference }
  const command = (executable: string, args: readonly string[], extra: Record<string, string> = {}) => Command.make(executable, ...args).pipe(
    Command.workingDirectory(root), Command.env({ ...environment, ...extra }), Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
    Effect.filterOrFail(code => code === 0, code => new AcceptanceFailed({ message: `Command exited ${code}: ${executable}` })), Effect.asVoid)
  const versions = ["0.0.501", "0.0.502", "0.0.503"] as const
  for (const version of versions) yield* command(process.execPath, [join(import.meta.dir, "build-desktop.ts")], {
    MAGNITUDE_ACCEPTANCE_VERSION: version, MAGNITUDE_ACCEPTANCE_OUTPUT: join(output, version), MAGNITUDE_ACCEPTANCE_CONFIG: config,
  })
  for (const replacement of [versions[1], versions[2]]) {
    const archive = join(output, replacement, "artifacts", windowsDesktopInstaller(replacement))
    const release = yield* signUpdateRelease({ version: replacement, bytes: Number((yield* fs.stat(archive)).size), sha256: yield* sha256File(archive) },
      { os: "windows", arch: "x64", package: "windows-exe" }, keys.privateKey)
    yield* fs.writeFileString(join(output, replacement, "prepared-release.json"), yield* Schema.encode(Schema.parseJson(UpdateRelease))(release))
  }
  yield* command(join(output, versions[0], "artifacts", windowsDesktopInstaller(versions[0])), ["/S"])
  for (const replacement of [versions[1], versions[2]]) {
    const archive = join(output, replacement, "artifacts", windowsDesktopInstaller(replacement))
    const release = yield* Schema.decodeUnknown(Schema.parseJson(UpdateRelease))(yield* fs.readFileString(join(output, replacement, "prepared-release.json")))
    const transfer = join(output, "transfer.exe")
    yield* fs.copyFile(archive, transfer)
    yield* Effect.gen(function* () {
      const store = yield* makePreparedUpdateStore({ dataDirectory, target: { os: "windows", arch: "x64", package: "windows-exe" }, trustedPublishers: new Map([["isolated", keys.publicKey]]) })
      yield* store.prepare(transfer, release)
    }).pipe(Effect.provide(windowsPrivateFilePermissions(nativeAddon)))
    if (replacement === versions[1]) yield* command(launcher, ["update", "install"])
    else yield* Effect.scoped(Effect.gen(function* () {
      const serving = yield* Command.make(launcher, "serve").pipe(Command.env(environment), Command.stdout("inherit"), Command.stderr("inherit"), Command.start)
      yield* Effect.gen(function* () {
        for (;;) {
          if (!(yield* serving.isRunning)) return yield* new AcceptanceFailed({ message: "Installed launcher exited before replacement became ready" })
          const status = yield* Command.make(launcher, "status").pipe(Command.env(environment), Command.string)
          if (/Runtime\s+Ready/.test(status) && /Owner\s+Headless/.test(status) && status.includes(replacement)) break
          yield* Effect.sleep("500 millis")
        }
      }).pipe(Effect.timeout("5 minutes"))
      const endpoint = yield* native.inspectEndpoint(stateDirectory)
      if (Option.isNone(endpoint)) return yield* new AcceptanceFailed({ message: "Replacement control endpoint is absent" })
      yield* requestApplication(endpoint.value, "Quit")
      if ((yield* serving.exitCode.pipe(Effect.timeout("30 seconds"))) !== 0) return yield* new AcceptanceFailed({ message: "Foreground launcher did not exit cleanly" })
    }))
    const version = (yield* Command.make(launcher, "--version").pipe(Command.string)).trim()
    const serviceVersion = (yield* Command.make(join(resources, "magnitude-service.exe"), "--version").pipe(Command.string)).trim()
    if (version !== replacement || serviceVersion !== replacement || (yield* fs.exists(join(dataDirectory, "updates/update.json")))) {
      return yield* new AcceptanceFailed({ message: "Replacement versions or prepared-state retirement differ" })
    }
  }
  yield* Effect.acquireUseRelease(command(launcher, ["app", "open"]), () => Effect.gen(function* () {
    yield* Effect.gen(function* () {
      for (;;) {
        const status = yield* Command.make(launcher, "status").pipe(Command.env(environment), Command.string)
        if (/Runtime\s+Ready/.test(status) && /Owner\s+Desktop/.test(status) && status.includes(versions[2])) break
        yield* Effect.sleep("500 millis")
      }
    }).pipe(Effect.timeout("5 minutes"))
    yield* command(launcher, ["models", "status"])
    yield* command(launcher, ["hardware"])
  }), () => Effect.gen(function* () {
    const endpoint = yield* native.inspectEndpoint(stateDirectory)
    if (Option.isSome(endpoint)) yield* requestApplication(endpoint.value, "Quit")
  }).pipe(Effect.orDie))
  yield* Effect.gen(function* () {
    for (;;) {
      const status = yield* Command.make(launcher, "status").pipe(Command.env(environment), Command.string)
      if (/Runtime\s+Stopped/.test(status)) break
      yield* Effect.sleep("100 millis")
    }
  }).pipe(Effect.timeout("30 seconds"))
  yield* fs.writeFileString(join(output, "result.json"), yield* Schema.encode(Schema.parseJson(Evidence))({ installedVersion: versions[2], finiteInstall: true, foregroundContinuation: true, gracefulExit: true, desktopCliRegression: true }))
  yield* Effect.logInfo("Windows signed finite and foreground update acceptance passed")
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
