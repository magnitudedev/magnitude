import { isWindows } from "../domain"
import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"
import { ICN_EXECUTABLE_NAME } from "@magnitudedev/release/executables"
import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { AssertionFailure } from "../domain"
import { ElfDependencyClosure, verifyElfDependencyClosure } from "../elf-dependency-closure"
import { ElfInterpreter, inspectElfInterpreter } from "../elf-interpreter"
import { InstalledApplication } from "../installer"
import { nativeInventory } from "../native-inventory"
import { admittedRuntimeComposition } from "../runtime-composition"
import { systemElfResolver } from "../system-elf"

// OS ABI dependencies of the Electron desktop and native host. CUDA toolkit and
// inference implementation libraries are deliberately absent: those must be owned.
export const linuxOsDependencies = ["libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1", "libresolv.so.2",
  "libgcc_s.so.1", "libstdc++.so.6", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1", "libz.so.1",
  "libglib-2.0.so.0", "libgobject-2.0.so.0", "libgio-2.0.so.0", "libgthread-2.0.so.0",
  "libnss3.so", "libnssutil3.so", "libsmime3.so", "libnspr4.so", "libplc4.so", "libplds4.so",
  "libdbus-1.so.3", "libudev.so.1", "libatk-1.0.so.0", "libatk-bridge-2.0.so.0", "libatspi.so.0", "libcups.so.2",
  "libdrm.so.2", "libX11.so.6", "libXcomposite.so.1", "libXdamage.so.1", "libXext.so.6", "libXfixes.so.3",
  "libXrandr.so.2", "libxcb.so.1", "libxkbcommon.so.0", "libgbm.so.1", "libasound.so.2",
  "libpango-1.0.so.0", "libcairo.so.2", "libgtk-3.so.0", "libgdk-3.so.0", "libexpat.so.1", "libfontconfig.so.1",
  "libfreetype.so.6", "libwayland-client.so.0", "libwayland-cursor.so.0", "libwayland-egl.so.1"] as const
export const LinuxPackageDependencies = Schema.Struct({ application: ElfDependencyClosure, runtime: ElfDependencyClosure,
  interpreters: Schema.Array(Schema.Struct({ file: Schema.String, interpreter: ElfInterpreter })), runtimeArtifacts: Schema.Array(Schema.String) })

export const inspectLinuxPackageDependencies = (app: InstalledApplication, release: typeof ReleaseManifestSchema.Type) => Effect.scoped(Effect.gen(function* () {
  const target = app.candidate.target
  if (target.os === "macos" || isWindows(target.os)) return yield* new AssertionFailure({ message: "ELF package inspection requires a Linux target" })
  const composed = yield* admittedRuntimeComposition(release, target)
  const fs = yield* FileSystem.FileSystem
  const interpreters: (typeof LinuxPackageDependencies.Type)["interpreters"][number][] = []
  const graph = (root: string, executable: string) => Effect.gen(function* () {
    const files = yield* nativeInventory(root)
    if (!files.length) return yield* new AssertionFailure({ message: `No native files in ${root}` })
    const entrypoint = yield* fs.realPath(executable)
    if (!files.some(file => file.path === entrypoint)) return yield* new AssertionFailure({ message: `Declared executable is absent from native inventory: ${executable}` })
    for (const file of files) {
      if (file.image.format !== "elf" || !file.image.architectures.includes(target.arch)) return yield* new AssertionFailure({ message: `Wrong native format or architecture: ${file.path}` })
      interpreters.push({ file: file.path, interpreter: yield* inspectElfInterpreter(file.path, target.arch) })
    }
    return yield* verifyElfDependencyClosure({ root, roots: files.map(file => file.path), arch: target.arch })
  })
  const resolver = systemElfResolver(target.packageFormat === "rpm" ? "rpm" : "deb",
    [...linuxOsDependencies, ...(target.backend === "cuda" ? ["libcuda.so.1"] : [])])
  // The package-manager launcher is a shell script; inspect the ELF payload it launches.
  const [application, runtime] = yield* Effect.all([graph(app.root, join(app.root, "magnitude")), graph(composed.root, join(composed.root, "bin", ICN_EXECUTABLE_NAME))]).pipe(Effect.provide(resolver))
  return LinuxPackageDependencies.make({ application, runtime, interpreters, runtimeArtifacts: composed.artifacts })
}))
