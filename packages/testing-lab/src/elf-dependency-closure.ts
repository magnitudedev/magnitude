import { FileSystem } from "@effect/platform"
import { Context, Effect, Schema } from "effect"
import { posix } from "node:path"
import { Architecture, AssertionFailure, InfrastructureFailure } from "./domain"
import { elfSearch } from "./elf-search"
import { inspectNativeDependencies } from "./native-dependencies"
import { attestElfVersions, inspectElfVersions } from "./elf-versions"
import { NvidiaDriverProvenance } from "./nvidia-driver-receipt"

export const SystemElf = Schema.Struct({ path: Schema.NonEmptyString, provenance: Schema.Union(
  Schema.Struct({ kind: Schema.Literal("package"), name: Schema.NonEmptyString }), NvidiaDriverProvenance,
) })
export interface SystemElfResolver {
  readonly resolve: (name: string, arch: typeof Architecture.Type) => Effect.Effect<typeof SystemElf.Type, AssertionFailure | InfrastructureFailure>
}
// A provider must verify native loader resolution and installation provenance. No fallback
// to a builder's arbitrary filesystem or unverified library-name allowlist is supplied.
export const SystemElfResolver = Context.GenericTag<SystemElfResolver>("@magnitudedev/testing-lab/SystemElfResolver")
export const ElfDependencyClosure = Schema.Struct({ root: Schema.String, files: Schema.Array(Schema.String),
  edges: Schema.Array(Schema.Struct({ from: Schema.String, name: Schema.String, resolution: Schema.Union(
    Schema.Struct({ kind: Schema.Literal("owned"), path: Schema.String }),
    Schema.Struct({ kind: Schema.Literal("system"), library: SystemElf }),
  ) })) })
const fail = (message: string) => new AssertionFailure({ message: `ELF dependency closure: ${message}` })

export const verifyElfDependencyClosure = (input: { readonly root: string; readonly roots: readonly string[];
  readonly arch: typeof Architecture.Type }) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const system = yield* SystemElfResolver
  const root = yield* fs.realPath(input.root)
  const inside = (path: string) => path.startsWith(root + "/")
  const owned = (path: string) => Effect.gen(function* () {
    const real = yield* fs.realPath(path)
    if (!inside(real) || (yield* fs.stat(real)).type !== "File") return yield* fail(`dependency escapes the package: ${path}`)
    return real
  })
  if (!input.roots.length) return yield* fail("no native roots supplied")
  const pending: { file: string; inherited: readonly string[] }[] = []
  for (const path of input.roots) pending.push({ file: yield* owned(path), inherited: [] })
  const visited = new Set<string>(), files = new Set<string>()
  const edges: (typeof ElfDependencyClosure.Type)["edges"][number][] = []
  for (let index = 0; index < pending.length; index++) {
    if (pending.length > 16_384) return yield* fail("dependency contexts exceed the inspection limit")
    const current = pending[index]!
    const key = yield* Schema.encode(Schema.parseJson(Schema.Tuple(Schema.String, Schema.Array(Schema.String))))([current.file, current.inherited])
    if (visited.has(key)) continue
    visited.add(key); files.add(current.file)
    const report = yield* inspectNativeDependencies(current.file, input.arch)
    if (report.format !== "elf") return yield* fail(`wrong native format: ${current.file}`)
    const versions = yield* inspectElfVersions(current.file)
    if (versions.needs.some(need => !report.imports.some(dependency => dependency.name === need.library))) {
      return yield* fail(`version requirement has no matching import in ${current.file}`)
    }
    const search = yield* elfSearch(root, current.file, report, current.inherited)
    for (const dependency of report.imports) {
      const name = dependency.name
      // A DT_NEEDED pathname bypasses normal search. Require an owned ORIGIN path;
      // neither absolute build paths nor cwd-relative paths may enter the package.
      const directPath = name.startsWith("$ORIGIN/") ? posix.resolve(posix.dirname(current.file), name.slice(8))
        : name.startsWith("${ORIGIN}/") ? posix.resolve(posix.dirname(current.file), name.slice(10)) : undefined
      if ((name.includes("/") || name.includes("$")) && (!directPath || !inside(directPath) || directPath.includes("$"))) {
        return yield* fail(`unowned DT_NEEDED pathname ${name} from ${current.file}`)
      }
      const candidates = directPath ? [directPath] : search.direct.map(directory => posix.join(directory, name))
      let selected: string | undefined
      for (const candidate of candidates) {
        if (yield* fs.exists(candidate)) { selected = yield* owned(candidate); break }
      }
      if (selected) {
        const required = versions.needs.filter(need => need.library === name).flatMap(need => need.versions)
        if (required.length) yield* attestElfVersions(name, required, yield* inspectElfVersions(selected))
        edges.push({ from: current.file, name, resolution: { kind: "owned", path: selected } })
        pending.push({ file: selected, inherited: search.inherited })
      } else {
        if (directPath) return yield* fail(`missing owned DT_NEEDED pathname ${name} from ${current.file}`)
        const library = yield* system.resolve(name, input.arch)
        if (!posix.isAbsolute(library.path) || inside(library.path)) return yield* fail(`invalid system resolution for ${name}`)
        const required = versions.needs.filter(need => need.library === name).flatMap(need => need.versions)
        if (required.length) yield* attestElfVersions(name, required, yield* inspectElfVersions(library.path))
        edges.push({ from: current.file, name, resolution: { kind: "system", library } })
      }
    }
  }
  return ElfDependencyClosure.make({ root, files: [...files].sort(), edges })
})
