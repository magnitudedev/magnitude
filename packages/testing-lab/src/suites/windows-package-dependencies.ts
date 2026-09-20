import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { ICN_EXECUTABLE_NAME } from "@magnitudedev/release/executables"
import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { join, win32 } from "node:path"
import { AssertionFailure, InfrastructureFailure } from "../domain"
import { InstalledApplication } from "../installer"
import { nativeInventory } from "../native-inventory"
import { PeDependencyClosure, verifyPeDependencyClosure } from "../pe-dependency-closure"
import { admittedRuntimeComposition } from "../runtime-composition"

export const WindowsPackageDependencies = Schema.Struct({ application: PeDependencyClosure,
  runtime: PeDependencyClosure, runtimeArtifacts: Schema.Array(Schema.String) })

export const inspectWindowsPackageDependencies = (app: InstalledApplication, release: typeof ReleaseManifestSchema.Type,
  environment: Readonly<Record<string, string>>) => Effect.scoped(Effect.gen(function* () {
  const target = app.candidate.target
  if (target.os !== "windows") return yield* new AssertionFailure({ message: "PE package inspection requires a Windows target" })
  const inspector = environment.LAB_DEPENDENCIES_EXECUTABLE, systemRoot = environment.SystemRoot ?? environment.SYSTEMROOT
  if (!inspector || !win32.isAbsolute(inspector) || !systemRoot || !win32.isAbsolute(systemRoot)) {
    return yield* new InfrastructureFailure({ operation: "pe-dependencies", message: "Pinned Windows dependency inspector or SystemRoot is unavailable" })
  }
  const fs = yield* FileSystem.FileSystem
  const composed = yield* admittedRuntimeComposition(release, target)
  const graph = (root: string, executable: string, runtime: boolean) => Effect.gen(function* () {
    const files = yield* nativeInventory(root), entrypoint = yield* fs.realPath(executable)
    if (!files.length || !files.some(file => file.path === entrypoint && file.image.architectures.includes(target.arch))) {
      return yield* new AssertionFailure({ message: `Missing matching native entrypoint: ${executable}` })
    }
    for (const file of files) {
      // Electron's installer utilities can be x86; each loader graph still requires
      // all resolved dependencies to match its own root. The inference graph is x64.
      if (file.image.format !== "pe" || (runtime ? !file.image.architectures.includes(target.arch)
        : !file.image.architectures.every(arch => arch === target.arch || arch === "x86"))) {
        return yield* new AssertionFailure({ message: `Wrong native package format or architecture: ${file.path}` })
      }
    }
    return yield* verifyPeDependencyClosure({ root, roots: files.map(file => file.path), inspector, systemRoot, cuda: target.backend === "cuda",
      ownedSearchPaths: runtime ? [join(root, "runtime")] : [] })
  })
  const application = yield* graph(app.root, app.executable, false)
  const runtime = yield* graph(composed.root, join(composed.root, "bin", `${ICN_EXECUTABLE_NAME}.exe`), true)
  return WindowsPackageDependencies.make({ application, runtime, runtimeArtifacts: composed.artifacts })
}))
