import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { join, resolve } from "node:path"
import { sha256File } from "../../src/macos-app"
import { UpdateManifest, signUpdateManifest } from "../../src/hosted-update/manifest"
import { writeInstallationDistribution } from "../build/installation-distribution"

class AcceptanceFixtureFailed extends Schema.TaggedError<AcceptanceFixtureFailed>()("AcceptanceFixtureFailed", { message: Schema.String }) {}
const run = Effect.gen(function* () {
  const [artifactInput, outputInput, version] = process.argv.slice(2)
  if (!artifactInput || !outputInput || !version || process.platform !== "linux") return yield* new AcceptanceFixtureFailed({ message: "Usage: prepare-linux-script-fixture.ts PACKAGE OUTPUT VERSION (on Linux)" })
  const artifact = resolve(artifactInput), output = resolve(outputInput)
  const fs = yield* FileSystem.FileSystem
  const packageType = artifact.endsWith(".deb") ? "deb" : artifact.endsWith(".rpm") ? "rpm" : undefined
  if (!packageType) return yield* new AcceptanceFixtureFailed({ message: "Expected a DEB or RPM package" })
  const arch = yield* Schema.decodeUnknown(Schema.Literal("arm64", "x64"))(process.arch)
  const filename = `magnitude.${packageType}`
  const manifest = yield* Schema.decodeUnknown(UpdateManifest)({ protocol: 1, version, tag: `@magnitudedev/cli@${version}`, commit: "a".repeat(40),
    artifact: { id: `desktop-linux-${arch}-${packageType}`, target: { os: "linux", arch, package: packageType }, filename,
      bytes: Number((yield* fs.stat(artifact)).size), sha256: yield* sha256File(artifact) } })
  const keys = yield* Effect.sync(() => generateKeyPairSync("ed25519"))
  const publication = yield* signUpdateManifest(manifest, keys.privateKey)
  yield* writeInstallationDistribution({ output, origin: "https://localhost:18443", appleTeam: "ABCDEFGHIJ", windowsPublisher: "Acceptance",
    publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString(), publications: [publication] })
  const directory = join(output, "magnitudedev/magnitude/releases/download", manifest.tag)
  yield* fs.makeDirectory(directory, { recursive: true })
  yield* fs.copyFile(artifact, join(directory, filename))
  yield* Effect.logInfo("Prepared local HTTPS installation fixture", { output })
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
