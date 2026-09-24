import { Command, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Schema } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { join, resolve } from "node:path"
import { appleSigning, signAppleCode } from "../apple/signing"
import { compileAppleBun } from "../apple/compile-bun"
import { desktopUpdateArchive } from "../../src/targets"
import { sha256File } from "../../src/macos-app"
import { signUpdateRelease } from "../../src/hosted-update/release"
import { makePreparedUpdateStore } from "../../../daemon-management/src/desktop-native/prepared-update"
import { unixPrivateFilePermissions } from "../../../daemon-management/src/desktop-native/private-files"

class AcceptanceFailed extends Schema.TaggedError<AcceptanceFailed>()("AcceptanceFailed", { message: Schema.String }) {}
const Configuration = Schema.Struct({ origin: Schema.String, keyId: Schema.String, publicKey: Schema.String })
const HarnessInvocation = Schema.Struct({ resources: Schema.String, stateDirectory: Schema.String, dataDirectory: Schema.String,
  version: Schema.String, architecture: Schema.Literal("arm64"), continuation: Schema.TaggedStruct("None", {}) })
const Evidence = Schema.Struct({ installedVersion: Schema.String, preparedRecordRetired: Schema.Literal(true),
  helperRetired: Schema.Literal(true), transactionRetired: Schema.Literal(true) })
const root = resolve(import.meta.dir, "../../../..")
const run = Effect.gen(function* () {
  if (process.platform !== "darwin" || process.arch !== "arm64") return yield* new AcceptanceFailed({ message: "Signed fixture requires a native macOS arm64 runner" })
  if ((yield* appleSigning).mode !== "developer-id") return yield* new AcceptanceFailed({ message: "This acceptance requires the production publisher signing path" })
  const fs = yield* FileSystem.FileSystem
  const output = resolve(yield* Config.string("MAGNITUDE_HEADLESS_ACCEPTANCE_OUTPUT"))
  yield* fs.makeDirectory(output, { mode: 0o700 })
  const keys = yield* Effect.sync(() => generateKeyPairSync("ed25519"))
  const config = join(output, "configuration.json")
  yield* fs.writeFileString(config, yield* Schema.encode(Schema.parseJson(Configuration))({ origin: "http://127.0.0.1:9", keyId: "isolated", publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString() }), { mode: 0o600 })
  const command = (executable: string, args: readonly string[], environment: Record<string, string> = {}) => Command.make(executable, ...args).pipe(
    Command.workingDirectory(root), Command.env(environment), Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
    Effect.filterOrFail(code => code === 0, code => new AcceptanceFailed({ message: `Acceptance command exited ${code}: ${executable}` })), Effect.asVoid)
  const versions = ["0.0.501", "0.0.502"] as const
  for (const version of versions) yield* command(process.execPath, [join(import.meta.dir, "build-desktop.ts")], {
    MAGNITUDE_ACCEPTANCE_VERSION: version, MAGNITUDE_ACCEPTANCE_OUTPUT: join(output, version),
    MAGNITUDE_ACCEPTANCE_CONFIG: config, MAGNITUDE_ACCEPTANCE_STANDARD_BUNDLE_ID: "true",
  })
  const installed = join(output, "installed")
  yield* fs.makeDirectory(installed, { mode: 0o700 })
  yield* command("/usr/bin/ditto", ["-x", "-k", join(output, versions[0], "artifacts", desktopUpdateArchive("darwin-arm64")), installed])
  const bundle = join(installed, "Magnitude.app"), resources = join(bundle, "Contents/Resources")
  const stateDirectory = join(output, "state"), dataDirectory = join(output, "profile")
  const archive = join(output, versions[1], "artifacts", desktopUpdateArchive("darwin-arm64"))
  const release = yield* signUpdateRelease({ version: versions[1], bytes: Number((yield* fs.stat(archive)).size), sha256: yield* sha256File(archive) },
    { os: "darwin", arch: "arm64", package: "mac-zip" }, keys.privateKey)
  const transfer = join(output, "transfer.zip")
  yield* fs.copyFile(archive, transfer)
  yield* Effect.gen(function* () {
    const store = yield* makePreparedUpdateStore({ dataDirectory, target: { os: "darwin", arch: "arm64", package: "mac-zip" }, trustedPublishers: new Map([["isolated", keys.publicKey]]) })
    yield* store.prepare(transfer, release)
  }).pipe(Effect.provide(unixPrivateFilePermissions))
  const harness = join(output, "installer-entry")
  yield* compileAppleBun(join(import.meta.dir, "mac-foreground-installer-entry.ts"), harness, "bun-darwin-arm64", "cli")
  yield* signAppleCode(harness, "dev.magnitude.installer-acceptance", "bun")
  yield* command(harness, [yield* Schema.encode(Schema.parseJson(HarnessInvocation))({ resources, stateDirectory, dataDirectory, version: versions[0], architecture: "arm64", continuation: { _tag: "None" } })])
  const actual = (yield* Command.make(join(resources, "magnitude"), "--version").pipe(Command.string)).trim()
  if (actual !== versions[1] || (yield* fs.exists(join(dataDirectory, "updates/update.json"))) ||
      (yield* fs.exists(join(installed, ".Magnitude.app.update"))) || (yield* fs.readDirectory(join(stateDirectory, "mac-installers"))).length) {
    return yield* new AcceptanceFailed({ message: "Installed version or transaction retirement did not match the signed update" })
  }
  yield* command("/usr/bin/codesign", ["--verify", "--deep", "--strict", bundle])
  yield* command("/usr/bin/xcrun", ["stapler", "validate", bundle])
  yield* command("/usr/sbin/spctl", ["--assess", "--type", "execute", "--verbose", bundle])
  yield* fs.writeFileString(join(output, "result.json"), yield* Schema.encode(Schema.parseJson(Evidence))({ installedVersion: actual, preparedRecordRetired: true, helperRetired: true, transactionRetired: true }))
  yield* Effect.logInfo("Signed finite macOS installer acceptance passed")
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
