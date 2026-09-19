import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { ReleaseArtifactSchema } from "@magnitudedev/release/contracts"
import { findTarget } from "../src/catalog"
import { TargetId } from "../src/domain"
import { InstalledApplication } from "../src/installer"
import { ProcessExecutor } from "../src/process"
import { inspectPackageIdentity } from "../src/suites/package"

for (const mode of ["valid", "wrong-desktop-version", "wrong-service-version", "wrong-cli-version", "wrong-architecture", "missing-host"] as const) test(`installed payload identity: ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-package-identity-" })
  const resources = join(root, "Contents", "Resources")
  yield* fs.makeDirectory(resources, { recursive: true })
  yield* fs.makeDirectory(join(root, "Contents", "MacOS"), { recursive: true })
  const target = yield* findTarget(TargetId.make("macos-15-arm64-metal-apple-silicon"))
  const artifact = yield* Schema.decodeUnknown(ReleaseArtifactSchema)({ id: "desktop-darwin-arm64", kind: "desktop", host: "darwin-arm64", filename: "Magnitude.dmg", bytes: 1, sha256: "a".repeat(64) })
  const executable = join(root, "Contents", "MacOS", "Magnitude")
  const app = InstalledApplication.make({ root, executable, cli: join(resources, "magnitude"), packageVersion: "0.1.3", candidate: { version: "0.1.3", artifact, target, path: "fixture.dmg" } })
  const header = new Uint8Array(64), view = new DataView(header.buffer)
  view.setUint32(0, 0xfeedfacf, true); view.setUint32(4, mode === "wrong-architecture" ? 0x01000007 : 0x0100000c, true)
  for (const path of [executable, ...["magnitude-service", "magnitude", "magnitude-command", ...(mode === "missing-host" ? [] : ["desktop-host.node"])].map(name => join(resources, name))]) yield* fs.writeFile(path, header)
  const commands: string[] = []
  const result = yield* inspectPackageIdentity(app, mode === "wrong-desktop-version" ? "9.9.9" : "0.1.3", { LAB_ISOLATED: "1" }).pipe(Effect.provideService(ProcessExecutor, { run: spec => Effect.sync(() => {
    commands.push(spec.executable)
    expect(spec.inheritEnv).toBe(false)
    expect(spec.env).toEqual({ LAB_ISOLATED: "1" })
    expect(spec.executable.startsWith(resources + "/")).toBe(true)
    const wrong = (mode === "wrong-service-version" && spec.args[0] === "version") || (mode === "wrong-cli-version" && spec.args[0] === "--version")
    return { exitCode: 0, stdout: wrong ? "9.9.9\n" : "0.1.3\n", stderr: "" }
  }) }), Effect.either)
  expect(result._tag).toBe(mode === "valid" ? "Right" : "Left")
  if (result._tag === "Right") {
    expect(result.right.binaries).toHaveLength(5)
    expect(commands).toEqual([join(resources, "magnitude-service"), join(resources, "magnitude")])
  }
  if (mode === "wrong-desktop-version" || mode === "wrong-architecture") expect(commands).toHaveLength(0)
})).pipe(Effect.provide(BunContext.layer))))
