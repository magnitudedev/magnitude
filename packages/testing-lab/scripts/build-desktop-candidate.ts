import { assertRuntime } from "../src/runtime"
import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Option, Schema } from "effect"
import { join, resolve } from "node:path"
import { buildAcnBinary } from "../../release/scripts/build/acn"
import { buildCliBinary } from "../../release/scripts/build/cli"
import { buildDesktopApplication, DesktopTarget } from "../../release/scripts/build/desktop"
import { buildDesktopDmg } from "../../release/scripts/apple/desktop"
import { buildLinuxDesktopInstaller } from "../../release/scripts/build/desktop-linux"
import { buildWindowsDesktopInstaller } from "../../release/scripts/build/desktop-windows"
import { ProcessExecutorLive, checkedCommand } from "../src/process"
import { InfrastructureFailure } from "../src/domain"
import { ACN_EXECUTABLE_NAME } from "../../release/src/executables"

// Invoked inside a disposable source workspace by the build worker. Release owns all package logic.
const root = resolve(import.meta.dir, "../../..")
const run = Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const output = resolve(yield* Config.string("LAB_BUILD_OUTPUT"))
  const target = yield* Schema.decodeUnknown(DesktopTarget)({ platform: process.platform, arch: process.arch })
  yield* fs.makeDirectory(output, { recursive: true })
  const execute = (args: readonly string[], cwd = root) => checkedCommand(process.execPath, args, {
    cwd: Option.some(cwd), timeoutMs: 30 * 60_000, maxOutputBytes: 32 * 1024 * 1024,
  })
  yield* execute(["packages/version/scripts/generate-version.ts"])
  const identity = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ cliVersion: Schema.String, revision: Schema.Int })))(
    yield* fs.readFileString(join(root, "packages/release/release-plan.json")))
  yield* execute(["run", "build"], join(root, "desktop"))
  const bunTarget = `bun-${target.platform === "win32" ? "windows" : target.platform}-${target.arch}`
  const service = yield* Effect.tryPromise({ try: () => buildAcnBinary(bunTarget), catch: () => new InfrastructureFailure({ operation: "build-acn", message: "Release-owned service compilation failed" }) })
  const cli = yield* Effect.tryPromise({ try: () => buildCliBinary(bunTarget), catch: () => new InfrastructureFailure({ operation: "build-cli", message: "Release-owned CLI compilation failed" }) })
  const apps = yield* buildDesktopApplication({ service, cli, version: identity.cliVersion, revision: identity.revision, outputDirectory: join(output, "application") })
  const app = target.platform === "darwin" ? join(apps[0]!, "Magnitude.app") : apps[0]!
  const resources = join(app, target.platform === "darwin" ? "Contents/Resources" : "resources")
  for (const [name, argument] of [[ACN_EXECUTABLE_NAME, "version"], ["magnitude", "--version"]] as const) {
    const observed = yield* checkedCommand(join(resources, `${name}${target.platform === "win32" ? ".exe" : ""}`), [argument])
    if (observed.stdout.trim() !== identity.cliVersion) return yield* new InfrastructureFailure({ operation: "package-identity", message: `${name} version does not match the package` })
  }
  if (target.platform === "darwin") yield* buildDesktopDmg({ app, output: join(output, "artifacts"), host: target.arch === "arm64" ? "darwin-arm64" : "darwin-x64" })
  else if (target.platform === "linux") {
    for (const format of ["deb", "rpm"] as const) yield* buildLinuxDesktopInstaller({ app, version: identity.cliVersion, revision: identity.revision,
      arch: target.arch, format, output: join(output, "artifacts") })
  } else {
    const guard = yield* Config.string("LAB_WINDOWS_INSTALL_GUARD")
    const makensis = yield* Config.string("LAB_WINDOWS_NSIS")
    yield* buildWindowsDesktopInstaller({ app, guard, makensis, version: identity.cliVersion, revision: identity.revision, output: join(output, "artifacts") })
  }
  yield* Effect.logInfo(`Packaged candidate: ${output}`)
})
BunRuntime.runMain(run.pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
