import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { ReleaseArtifactSchema } from "@magnitudedev/release/contracts"
import { Config, Effect, Layer, Schema, Stream } from "effect"
import { createHash } from "node:crypto"
import { join } from "node:path"
import { Candidate } from "../src/candidate"
import { findTarget } from "../src/catalog"
import { TargetId, AssertionFailure } from "../src/domain"
import { Installer, nativeInstaller } from "../src/installer"
import { checkedCommand, ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"

const run = Effect.gen(function* () {
  yield* assertRuntime
  const file = yield* Config.string("LAB_INSTALL_PACKAGE")
  const version = yield* Config.string("LAB_INSTALL_VERSION")
  const root = yield* Config.string("LAB_INSTALL_ROOT")
  const target = yield* findTarget(TargetId.make(yield* Config.string("LAB_INSTALL_TARGET")))
  const disposable = yield* Config.boolean("LAB_INSTALL_DISPOSABLE").pipe(Config.withDefault(false))
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(root, { recursive: true })
  const hash = createHash("sha256")
  let bytes = 0
  yield* fs.stream(file).pipe(Stream.runForEach(chunk => Effect.sync(() => { hash.update(chunk); bytes += chunk.byteLength })))
  const artifact = yield* Schema.decodeUnknown(ReleaseArtifactSchema)({ id: `desktop-${target.artifactHost}`, kind: "desktop", host: target.artifactHost,
    filename: `Magnitude.${target.packageFormat}`, sha256: hash.digest("hex"), bytes })
  const candidate = Candidate.make({ artifact, version, target, path: file })
  const environment = Object.fromEntries(["HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "LOCALAPPDATA", "APPDATA", "SystemRoot", "TEMP"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const program = Effect.gen(function* () {
    const installer = yield* Installer
    const app = yield* installer.install(candidate)
    yield* fs.writeFileString(join(root, "installed.json"), yield* Schema.encode(Schema.parseJson(Schema.Unknown))(app))
    const result = yield* checkedCommand(app.cli, ["--version"], { env: environment, inheritEnv: false })
    yield* fs.writeFileString(join(root, "cli-version.txt"), result.stdout)
    if (result.stdout.trim() !== version) return yield* new AssertionFailure({ message: "Bundled CLI did not report expected version" })
    yield* installer.uninstall(app)
    yield* fs.writeFileString(join(root, "install-report.json"), '{"installed":true,"bundledCliVersion":true,"removed":true}\n')
  }).pipe(Effect.provide(nativeInstaller({ disposable, root: join(root, "application"), environment })))
  yield* program
})
BunRuntime.runMain(run.pipe(Effect.provide(Layer.merge(BunContext.layer, ProcessExecutorLive))))
