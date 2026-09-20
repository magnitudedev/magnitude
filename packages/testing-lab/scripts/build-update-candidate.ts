import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Option, Schema } from "effect"
import { join, resolve } from "node:path"
import { ReleaseArtifactSchema, ReleaseManifestSchema, validateReleaseManifest } from "@magnitudedev/release/contracts"
import { acnArchive, currentHost } from "../../release/src/targets"
import { ACN_EXECUTABLE_NAME } from "../../release/src/executables"
import { buildArchive } from "../../release/scripts/build/common"
import { buildMacApp } from "../../release/scripts/apple/build-app"
import { regularAppleFiles } from "../../release/scripts/apple/distribution"
import { checkedCommand, ProcessExecutorLive } from "../src/process"
import { InfrastructureFailure } from "../src/domain"

/** Release owns application compilation and installers; the lab assembles the exact fixture graph. */
const program = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = resolve(import.meta.dir, "../../..")
  const normal = resolve(yield* Config.string("LAB_BUILD_OUTPUT"))
  const output = resolve(yield* Config.string("MAGNITUDE_ACCEPTANCE_OUTPUT"))
  const version = yield* Config.string("MAGNITUDE_ACCEPTANCE_VERSION")
  const original = yield* fs.readFileString(join(normal, "artifacts/release-manifest.json")).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(ReleaseManifestSchema))))
  yield* checkedCommand(process.execPath, ["packages/release/scripts/acceptance/build-desktop.ts"], {
    cwd: Option.some(root), timeoutMs: 60 * 60_000, maxOutputBytes: 32 * 1024 * 1024,
  })
  const host = currentHost(), directory = join(output, "artifacts")
  const desktop = yield* Effect.forEach((yield* fs.readDirectory(directory)).filter(name => name.endsWith(".artifact.json")), name =>
    fs.readFileString(join(directory, name)).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(ReleaseArtifactSchema)))))
  if (!desktop.length || desktop.some(artifact => artifact.kind !== "desktop")) return yield* new InfrastructureFailure({ operation: "update-build", message: "Acceptance packaging did not produce desktop installers" })
  const appRoot = join(output, "application", `Magnitude-${process.platform}-${process.arch}`)
  const app = process.platform === "darwin" ? join(appRoot, "Magnitude.app") : appRoot
  const service = join(app, process.platform === "darwin" ? "Contents/Resources" : "resources", `${ACN_EXECUTABLE_NAME}${process.platform === "win32" ? ".exe" : ""}`)
  const sources = process.platform === "darwin"
    ? (yield* regularAppleFiles(yield* buildMacApp(join(output, "service-application"), service, version, original.acnRevision))).map(file => ({ ...file, path: `Magnitude.app/${file.path}` }))
    : [{ path: `bin/${ACN_EXECUTABLE_NAME}${process.platform === "win32" ? ".exe" : ""}`, source: service, mode: 0o755 }]
  const archive = yield* Effect.tryPromise({ try: () => buildArchive(join(directory, acnArchive(host)), join(directory, `acn-${host}.artifact.json`), {
    id: `acn-${host}`, kind: "acn", host: Option.some(host), backend: Option.none(), requiredBaseId: Option.none(),
    nativeBuild: Option.none(), backendModuleAbi: Option.none(), compatibility: Option.none(),
  }, sources), catch: () => new InfrastructureFailure({ operation: "update-build", message: "Acceptance service archive failed" }) })
  const native = original.artifacts.filter(artifact => artifact.kind === "icn-base" || artifact.kind === "icn-backend")
  for (const artifact of native) yield* fs.copyFile(join(normal, "artifacts", artifact.filename), join(directory, artifact.filename))
  const release = yield* validateReleaseManifest(ReleaseManifestSchema.make({ ...original, version, tag: `@magnitudedev/cli@${version}`, artifacts: [archive, ...desktop, ...native] }))
  yield* fs.writeFileString(join(directory, "release-manifest.json"), yield* Schema.encode(Schema.parseJson(ReleaseManifestSchema))(release), { flag: "wx" })
})
BunRuntime.runMain(program.pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
