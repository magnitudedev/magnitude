import { Command, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Schema } from "effect"
import { join, resolve } from "node:path"
import { buildAcnBinary } from "../build/acn"
import { buildDesktopApplication } from "../build/desktop"
import { buildDesktopDmg } from "../apple/desktop"
import { buildLinuxDesktopInstaller } from "../build/desktop-linux"
import { ACN_EXECUTABLE_NAME } from "../../src/executables"

class AcceptanceBuildFailed extends Schema.TaggedError<AcceptanceBuildFailed>()("AcceptanceBuildFailed", { message: Schema.String }) {}
const root = resolve(import.meta.dir, "../../../..")
const Package = Schema.Record({ key: Schema.String, value: Schema.Unknown })
const run = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const version = yield* Config.literal("0.0.14", "0.0.15")("MAGNITUDE_ACCEPTANCE_VERSION")
  const target = yield* Schema.decodeUnknown(Schema.Union(
    Schema.Struct({ platform: Schema.Literal("darwin"), arch: Schema.Literal("arm64") }),
    Schema.Struct({ platform: Schema.Literal("linux"), arch: Schema.Literal("arm64", "x64") }),
  ))({ platform: process.platform, arch: process.arch })
  const output = resolve(yield* Config.string("MAGNITUDE_ACCEPTANCE_OUTPUT"))
  yield* fs.makeDirectory(output, { recursive: true })
  const configPath = join(output, "update-acceptance.json")
  yield* fs.writeFileString(configPath, yield* Schema.encode(Schema.parseJson(Schema.Struct({ origin: Schema.String, storageOrigin: Schema.String, keyId: Schema.String, publicKey: Schema.String })) )({
    origin: "https://magnitude-update-acceptance.vercel.app", storageOrigin: "https://5r3lqtpag4uzvtxd.public.blob.vercel-storage.com",
    keyId: "acceptance", publicKey: yield* fs.readFileString(join(root, "packages/release/resources/distribution/acceptance.pub.pem")),
  }))
  const command = (args: readonly [string, ...string[]], cwd = root) => Command.make(...args).pipe(Command.workingDirectory(cwd),
    Command.env({ MAGNITUDE_UPDATE_ACCEPTANCE_CONFIG: configPath }), Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
    Effect.flatMap(code => code === 0 ? Effect.void : new AcceptanceBuildFailed({ message: `${args[0]} exited ${code}` })))
  const packagePath = join(root, "packages/launcher/package.json")
  yield* Effect.acquireUseRelease(fs.readFileString(packagePath), original => Effect.gen(function* () {
    const packageJson = yield* Schema.decodeUnknown(Schema.parseJson(Package))(original)
    yield* fs.writeFileString(packagePath, yield* Schema.encode(Schema.parseJson(Package))({ ...packageJson, version }))
    yield* command([process.execPath, "packages/version/scripts/generate-version.ts"])
    yield* command([process.execPath, "run", "build"], join(root, "desktop"))
    const service = yield* Effect.tryPromise({ try: () => buildAcnBinary(`bun-${target.platform}-${target.arch}`), catch: () => new AcceptanceBuildFailed({ message: "Service compilation failed" }) })
    const release = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ revision: Schema.Number })))(yield* fs.readFileString(join(root, "packages/release/release-plan.json")))
    const apps = yield* buildDesktopApplication({ service, version, revision: release.revision, outputDirectory: join(output, "application") })
    const app = target.platform === "darwin" ? join(apps[0]!, "Magnitude.app") : apps[0]!
    const serviceVersion = yield* Command.make(join(app, target.platform === "darwin" ? "Contents/Resources" : "resources", ACN_EXECUTABLE_NAME), "version").pipe(Command.string)
    if (serviceVersion.trim() !== version) return yield* new AcceptanceBuildFailed({ message: "Application and bundled service versions differ" })
    if (target.platform === "darwin") {
      // Separate Launch Services identity; the executable, service and native installation path are real.
      yield* command(["/usr/libexec/PlistBuddy", "-c", "Set :CFBundleIdentifier dev.magnitude.desktop.update-acceptance", join(app, "Contents/Info.plist")])
      yield* buildDesktopDmg({ app, output: join(output, "artifacts"), host: "darwin-arm64" })
    } else {
      for (const format of ["deb", "rpm"] as const) yield* buildLinuxDesktopInstaller({ app, version, revision: release.revision, arch: target.arch, format, output: join(output, "artifacts") })
    }
  }), original => fs.writeFileString(packagePath, original).pipe(Effect.orDie))
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
