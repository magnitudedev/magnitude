import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { Effect, Layer, Schema, Stream } from "effect"
import { basename, dirname, join } from "node:path"
import { expect, test } from "vitest"
import releasePlan from "../../release/release-plan.json"
import { NodeArchiveExtractor } from "../../release/src/archive"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { targets } from "../src/catalog"
import { InstalledApplication } from "../src/installer"
import { checkedCommand, ProcessExecutor, ProcessExecutorLive } from "../src/process"
import { sha256 } from "../src/snapshot"
import { inspectWindowsPackageTrust, WindowsPublisher } from "../src/suites/windows-package-trust"

for (const mode of ["owned", "missing-service", "non-native-service", "misplaced-microsoft"] as const) test(`Windows trust inventory: ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "lab-windows-trust-" })
  const header = new Uint8Array(256), view = new DataView(header.buffer)
  view.setUint16(0, 0x5a4d, true); view.setUint32(60, 128, true); view.setUint32(128, 0x00004550, true)
  view.setUint16(132, 0x8664, true); view.setUint16(152, 0x20b, true)
  const write = (path: string) => fs.makeDirectory(dirname(path), { recursive: true }).pipe(Effect.zipRight(fs.writeFile(path, header)))
  const appRoot = join(temporary, "app"), runtime = join(temporary, "runtime")
  for (const path of ["Magnitude.exe", "resources/magnitude.exe", "resources/magnitude-service.exe", "resources/desktop-host.node", "Uninstall Magnitude.exe"]) yield* write(join(appRoot, path))
  if (mode === "missing-service") yield* fs.remove(join(appRoot, "resources/magnitude-service.exe"))
  if (mode === "non-native-service") yield* fs.writeFileString(join(appRoot, "resources/magnitude-service.exe"), "not native code")
  const runtimeFiles = ["bin/magnitude-inference.exe", "backends/ggml-cpu.dll", "runtime/vcruntime140.dll",
    ...(mode === "misplaced-microsoft" ? ["backends/vcruntime140.dll"] : [])]
  for (const path of runtimeFiles) yield* write(join(runtime, path))
  yield* fs.makeDirectory(join(runtime, "catalog"))
  yield* fs.writeFileString(join(runtime, "catalog/model-planner-inputs.bundle"), "catalog fixture")
  const archive = join(temporary, "base.tar.gz")
  yield* checkedCommand("tar", ["-czf", archive, "-C", runtime, ...runtimeFiles, "catalog/model-planner-inputs.bundle"], { env: { COPYFILE_DISABLE: "1" } })
  const bytes = yield* fs.readFile(archive)
  const release = yield* Schema.decodeUnknown(ReleaseManifestSchema)({ schemaVersion: 2, version: "0.1.3", acnRevision: 1,
    rpc: releasePlan.rpc, plugins: [], tag: "@magnitudedev/cli@0.1.3", sourceCommit: "a".repeat(40), artifacts: [
      { id: "base", kind: "icn-base", host: "windows-x64-msvc", backend: "cpu", nativeBuild: "fixture", backendModuleAbi: "fixture", filename: "base.tar.gz", bytes: bytes.length, sha256: sha256(bytes) },
    ] })
  const app = yield* Schema.decodeUnknown(InstalledApplication)({ root: appRoot, executable: join(appRoot, "Magnitude.exe"), cli: join(appRoot, "resources/magnitude.exe"), packageVersion: "0.1.3",
    candidate: { version: "0.1.3", path: join(temporary, "installer.exe"), target: targets.find(target => target.id === "windows-server-2025-x64-cpu-intel"),
      artifact: { id: "desktop", kind: "desktop", host: "windows-x64-msvc", filename: "installer.exe", bytes: 256, sha256: sha256(header) } } })
  yield* write(app.candidate.path)
  const signatures = Layer.succeed(ProcessExecutor, { run: spec => Effect.succeed({ exitCode: 0, stderr: "", stdout: spec.executable === "signtool.exe" ? "" : JSON.stringify({ status: "Valid", signatureType: "Authenticode", timestamped: true,
    publisher: basename(spec.env.LAB_SIGNATURE_PATH!) === "vcruntime140.dll" ? "Microsoft Windows Software Compatibility Publisher" : "Magnitude Fixture" }) }) })
  yield* Effect.gen(function* () {
    yield* (yield* ArtifactStore).put(sha256(bytes), Stream.make(bytes))
    const result = yield* inspectWindowsPackageTrust(app, release, { kind: "production", publisher: WindowsPublisher.make("Magnitude Fixture"), signtool: "signtool.exe" }, {}).pipe(Effect.provide(signatures), Effect.either)
    expect(result._tag, result._tag === "Left" ? String(result.left) : "").toBe(mode === "owned" ? "Right" : "Left")
    if (result._tag === "Right") {
      expect(result.right.signatures).toHaveLength(9)
      expect(result.right.signatures.map(signature => signature.path)).toContain("runtime/runtime/vcruntime140.dll")
    } else {
      expect(String(result.left)).toContain(mode === "missing-service" ? "Required native file is missing"
        : mode === "non-native-service" ? "not in the native inventory" : "not yet independently verified")
    }
  }).pipe(Effect.provide(fileArtifactStore(join(temporary, "objects"))))
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive, NodeArchiveExtractor]))))
