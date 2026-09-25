import { Command, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Schema } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { join, resolve } from "node:path"
import { windowsDesktopInstaller, cliArchive } from "../../src/targets"
import { sha256File } from "../../src/macos-app"
import { UpdateManifest, signUpdateManifest } from "../../src/hosted-update/manifest"
import { writeInstallationDistribution } from "../build/installation-distribution"

class AcceptanceFailed extends Schema.TaggedError<AcceptanceFailed>()("AcceptanceFailed", { message: Schema.String }) {}
const Configuration = Schema.Struct({ origin: Schema.String, keyId: Schema.String, publicKey: Schema.String, windowsPublisher: Schema.String })
const run = Effect.gen(function* () {
  if (process.platform !== "win32") return yield* new AcceptanceFailed({ message: "Requires a Windows test machine" })
  const fs = yield* FileSystem.FileSystem
  const output = resolve(yield* Config.string("MAGNITUDE_SCRIPT_ACCEPTANCE_OUTPUT"))
  yield* fs.makeDirectory(output, { recursive: true })
  const version = "0.0.505"
  const keys = yield* Effect.sync(() => generateKeyPairSync("ed25519"))
  const publicKey = keys.publicKey.export({ type: "spki", format: "pem" }).toString()
  const configPath = join(output, "configuration.json")
  yield* fs.writeFileString(configPath, yield* Schema.encode(Schema.parseJson(Configuration))({ origin: "http://127.0.0.1:9", keyId: "acceptance",
    publicKey, windowsPublisher: "Magnitude Update Acceptance" }))
  const build = join(output, "build")
  const code = yield* Command.make(process.execPath, join(import.meta.dir, "build-desktop.ts")).pipe(
    Command.env({ MAGNITUDE_ACCEPTANCE_VERSION: version, MAGNITUDE_ACCEPTANCE_OUTPUT: build, MAGNITUDE_ACCEPTANCE_CONFIG: configPath }),
    Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode)
  if (code !== 0) return yield* new AcceptanceFailed({ message: "Signed package compilation failed" })
  const filename = windowsDesktopInstaller(version)
  const installer = join(build, "artifacts", filename)
  const manifest = yield* Schema.decodeUnknown(UpdateManifest)({ protocol: 1, version, tag: `@magnitudedev/cli@${version}`, commit: "a".repeat(40),
    artifact: { id: "desktop-windows-x64-msvc", target: { os: "windows", arch: "x64", package: "windows-exe" }, filename,
      bytes: Number((yield* fs.stat(installer)).size), sha256: yield* sha256File(installer) } })
  const publication = yield* signUpdateManifest(manifest, keys.privateKey)
  const hosting = join(output, "hosting")
  yield* writeInstallationDistribution({ output: hosting, origin: "https://localhost:18443", appleTeam: "ABCDEFGHIJ",
    windowsPublisher: "Magnitude Update Acceptance", publicKey, publications: [publication] })
  const artifacts = join(hosting, "magnitudedev/magnitude/releases/download", manifest.tag)
  yield* fs.makeDirectory(artifacts, { recursive: true })
  yield* fs.copyFile(installer, join(artifacts, filename))
  const applications = yield* fs.readDirectory(join(build, "application"))
  if (applications.length !== 1) return yield* new AcceptanceFailed({ message: "Expected one matched application" })
  const archiveRoot = join(output, "cli")
  yield* fs.makeDirectory(join(archiveRoot, "bin"), { recursive: true })
  yield* fs.copyFile(join(build, "application", applications[0]!, "resources/magnitude.exe"), join(archiveRoot, "bin/magnitude-cli.exe"))
  if ((yield* Command.make("tar.exe", "-czf", join(artifacts, cliArchive("windows-x64-msvc")), "-C", archiveRoot, "bin/magnitude-cli.exe").pipe(Command.exitCode)) !== 0) {
    return yield* new AcceptanceFailed({ message: "Signed verifier packaging failed" })
  }
  yield* Effect.logInfo("Prepared signed Windows script fixture", { hosting })
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
