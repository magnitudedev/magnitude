import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { FileSystem } from "@effect/platform"
import { ICN_EXECUTABLE_NAME } from "@magnitudedev/release/executables"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { AssertionFailure } from "../domain"
import { InstalledApplication } from "../installer"
import { MachDependencyClosure, verifyMachDependencyClosure } from "../mach-dependency-closure"
import { machExecutable, nativeInventory } from "../native-inventory"
import { admittedRuntimeComposition } from "../runtime-composition"

export const MacPackageDependencies = Schema.Struct({ application: Schema.Array(MachDependencyClosure),
  runtime: MachDependencyClosure, runtimeArtifacts: Schema.Array(Schema.String) })

/** Cover every packaged executable and library, including libraries opened dynamically. */
export const inspectMacPackageDependencies = (app: InstalledApplication, release: typeof ReleaseManifestSchema.Type) => Effect.scoped(Effect.gen(function* () {
  const target = app.candidate.target
  if (target.os !== "macos") return yield* new AssertionFailure({ message: "Mach-O package verification requires a macOS target" })
  const applicationFiles = yield* nativeInventory(app.root)
  if (applicationFiles.length === 0) return yield* new AssertionFailure({ message: "Installed application contains no native files" })
  const executables: string[] = [], libraries: string[] = []
  for (const file of applicationFiles) {
    if (file.image.format !== "mach-o" || !file.image.architectures.includes(target.arch)) {
      return yield* new AssertionFailure({ message: `Packaged native file has the wrong format or architecture: ${file.path}` })
    }
    const executable = yield* machExecutable(file.path, target.arch)
    if (executable) executables.push(file.path)
    else libraries.push(file.path)
  }
  const roots = new Map(executables.map(path => [path, [path]]))
  // Nested helper apps have their own executable-relative loader context. Other
  // libraries belong to the main application, including dynamically loaded bridges.
  for (const library of libraries) {
    const appBoundary = library.lastIndexOf(".app/Contents/")
    const ownerRoot = appBoundary >= 0 ? library.slice(0, appBoundary + 4) : app.root
    const candidates = executables.filter(path => path.startsWith(`${ownerRoot}/Contents/MacOS/`))
    if (candidates.length !== 1) return yield* new AssertionFailure({ message: `No unique application loader context for ${library}` })
    roots.get(candidates[0]!)!.push(library)
  }
  const application = yield* Effect.forEach([...roots], ([executable, files]) => verifyMachDependencyClosure({ root: app.root,
    executable, roots: files, arch: target.arch }))
  const composed = yield* admittedRuntimeComposition(release, target)
  const fs = yield* FileSystem.FileSystem
  const runtimeExecutable = yield* fs.realPath(join(composed.root, "bin", ICN_EXECUTABLE_NAME))
  const runtimeFiles = yield* nativeInventory(composed.root)
  if (runtimeFiles.length === 0) return yield* new AssertionFailure({ message: "Admitted runtime contains no native files" })
  for (const file of runtimeFiles) {
    if (file.image.format !== "mach-o" || !file.image.architectures.includes(target.arch)) {
      return yield* new AssertionFailure({ message: `Runtime native file has the wrong format or architecture: ${file.path}` })
    }
    if ((yield* machExecutable(file.path, target.arch)) !== (file.path === runtimeExecutable)) {
      return yield* new AssertionFailure({ message: `Runtime executable loader context differs from the declared binary: ${file.path}` })
    }
  }
  const runtime = yield* verifyMachDependencyClosure({ root: composed.root, executable: runtimeExecutable,
    roots: runtimeFiles.map(file => file.path), arch: target.arch })
  return MacPackageDependencies.make({ application, runtime, runtimeArtifacts: composed.artifacts })
}))
