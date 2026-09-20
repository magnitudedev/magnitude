import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import releasePlan from "../../release/release-plan.json"
import { NodeArchiveExtractor } from "../../release/src/archive"
import { fileArtifactStore } from "../src/artifact-store"
import { Candidate } from "../src/candidate"
import { targets } from "../src/catalog"
import { InstalledApplication } from "../src/installer"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { checkedCommand, ProcessExecutorLive } from "../src/process"
import { inspectDebianPackageTrust, inspectDebianSignature } from "../src/suites/debian-package-trust"

for (const mode of ["unsigned", "signature-present", "unknown-extension", "missing-control", "malformed-container"] as const) test.skipIf(process.platform === "win32")(`DEB signature inspection fails closed for ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-deb-signature-" })
  const archive = join(root, "fixture.deb")
  if (mode === "malformed-container") yield* fs.writeFileString(archive, "invalid package")
  else {
    const members = ["debian-binary", ...(mode === "missing-control" ? [] : ["control.tar.gz"]), "data.tar.gz",
      ...(mode === "signature-present" ? ["_gpgorigin"] : mode === "unknown-extension" ? ["unknown"] : [])]
    // These real ar fixtures qualify signature-member classification only, not DEB payload validity.
    for (const member of members) yield* fs.writeFileString(join(root, member), member === "debian-binary" ? "2.0\n" : "fixture")
    yield* checkedCommand("ar", ["qc", archive, ...members], { cwd: Option.some(root) })
  }
  const result = yield* inspectDebianSignature(archive).pipe(Effect.either)
  expect(result._tag).toBe(mode === "unsigned" ? "Right" : "Left")
  if (result._tag === "Right") expect(result.right).toBe("Unsigned")
  if (mode === "signature-present" && result._tag === "Left") {
    expect(result.left._tag).toBe("InfrastructureFailure")
    expect(result.left.message).toContain("publisher")
  }
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))

// A development receipt must never make the release profile pass without a publisher policy.
test("production DEB trust stays blocked even before reading an otherwise admitted package", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-deb-production-policy-" })
  const target = targets.find(value => value.id === "ubuntu-24.04-x64-cpu-intel")!
  const artifact = { id: "desktop-linux-x64", kind: "desktop", host: "linux-x64-gnu", filename: "Magnitude.deb", bytes: 1, sha256: "a".repeat(64) }
  const candidate = yield* Schema.decodeUnknown(Candidate)({ target, version: "0.1.3", path: join(root, "not-read.deb"), artifact })
  const release = yield* Schema.decodeUnknown(ReleaseManifestSchema)({ schemaVersion: 2, version: "0.1.3", acnRevision: 1,
    rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.3", sourceCommit: "b".repeat(40), artifacts: [artifact] })
  const app = InstalledApplication.make({ candidate, root, executable: join(root, "app"), cli: join(root, "cli"), packageVersion: "0.1.3" })
  const result = yield* inspectDebianPackageTrust(app, release, true).pipe(Effect.provide([fileArtifactStore(join(root, "objects")), NodeArchiveExtractor]), Effect.either)
  expect(result._tag).toBe("Left")
  if (result._tag === "Left") {
    expect(result.left._tag).toBe("InfrastructureFailure")
    expect(result.left.message).toContain("production trust")
  }
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
