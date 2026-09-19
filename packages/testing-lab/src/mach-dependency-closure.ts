import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { dirname, isAbsolute, normalize, resolve, sep } from "node:path"
import { Architecture, AssertionFailure } from "./domain"
import { inspectNativeDependencies } from "./native-dependencies"

const Edge = Schema.Struct({ from: Schema.NonEmptyString, name: Schema.NonEmptyString,
  linkage: Schema.Literal("required", "weak", "delay"), resolution: Schema.Union(
    Schema.Struct({ kind: Schema.Literal("owned"), path: Schema.NonEmptyString }),
    Schema.Struct({ kind: Schema.Literal("system"), path: Schema.NonEmptyString }),
  ) })
export const MachDependencyClosure = Schema.Struct({ root: Schema.NonEmptyString,
  executable: Schema.NonEmptyString, files: Schema.Array(Schema.NonEmptyString), edges: Schema.Array(Edge) })
const failed = (message: string) => new AssertionFailure({ message: `Mach-O dependency closure: ${message}` })
// System images may live only in the dyld shared cache. Record the OS boundary rather
// than pretend that stat() or a basename search proves the contents of that cache.
const system = (path: string) => normalize(path) === path && (path.startsWith("/usr/lib/") || path.startsWith("/System/Library/"))

/** Resolve one executable's owned graph using its actual load-command ancestry. */
export const verifyMachDependencyClosure = (input: { readonly root: string; readonly executable: string;
  readonly roots: readonly string[]; readonly arch: typeof Architecture.Type }) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.realPath(input.root)
  const inside = (path: string) => path.startsWith(root + sep)
  const owned = (path: string) => Effect.gen(function* () {
    const canonical = yield* fs.realPath(path)
    if (!inside(canonical) || (yield* fs.stat(canonical)).type !== "File") return yield* failed(`dependency is not an owned regular file: ${path}`)
    return canonical
  })
  const executable = yield* owned(resolve(input.executable))
  if (input.roots.length === 0) return yield* failed("no native roots were supplied")
  const initialRoots = yield* Effect.forEach(input.roots, path => owned(resolve(path)))
  const pending: { file: string; inherited: readonly string[] }[] = [{ file: executable, inherited: [] }]
  const files = new Set<string>(), visited = new Set<string>(), edges: (typeof Edge.Type)[] = []
  const declarations = new Map<string, Effect.Effect.Success<ReturnType<typeof inspectNativeDependencies>>>()
  for (let index = 0; index < pending.length; index++) {
    if (pending.length > 16_384) return yield* failed("dependency graph exceeds its context limit")
    const current = pending[index]!
    const key = yield* Schema.encode(Schema.parseJson(Schema.Tuple(Schema.String, Schema.Array(Schema.String))))([current.file, current.inherited])
    if (visited.has(key)) continue
    visited.add(key)
    files.add(current.file)
    let report = declarations.get(current.file)
    if (!report) { report = yield* inspectNativeDependencies(current.file, input.arch); declarations.set(current.file, report) }
    if (report.format !== "mach-o") return yield* failed(`non-Mach-O dependency: ${current.file}`)
    const expand = (path: string) => Effect.gen(function* () {
      const expanded = path === "@loader_path" ? dirname(current.file)
        : path.startsWith("@loader_path/") ? resolve(dirname(current.file), path.slice(13))
        : path === "@executable_path" ? dirname(executable)
        : path.startsWith("@executable_path/") ? resolve(dirname(executable), path.slice(17))
        : isAbsolute(path) && system(path) ? path : undefined
      if (!expanded || (!inside(expanded) && expanded !== root && !system(expanded))) return yield* failed(`unowned or relative loader path in ${current.file}: ${path}`)
      return expanded
    })
    const localPaths = yield* Effect.forEach(report.rpaths, expand)
    const search = [...new Set([...localPaths, ...current.inherited])]
    if (index === 0) for (const file of initialRoots) {
      if (file !== executable) pending.push({ file, inherited: search })
    }
    for (const dependency of report.imports) {
      if (system(dependency.name)) {
        edges.push({ from: current.file, ...dependency, resolution: { kind: "system", path: dependency.name } })
        continue
      }
      const candidates = dependency.name.startsWith("@rpath/")
        ? search.map(path => resolve(path, dependency.name.slice(7))) : [yield* expand(dependency.name)]
      let selected: string | undefined
      for (const candidate of candidates) {
        if (!inside(candidate)) return yield* failed(`dependency search escapes the owned root: ${dependency.name} from ${current.file}`)
        if (yield* fs.exists(candidate)) { selected = yield* owned(candidate); break }
      }
      if (!selected) return yield* failed(`unresolved ${dependency.linkage} import ${dependency.name} from ${current.file}`)
      edges.push({ from: current.file, ...dependency, resolution: { kind: "owned", path: selected } })
      pending.push({ file: selected, inherited: search })
    }
  }
  return MachDependencyClosure.make({ root, executable, files: [...files].sort(), edges })
})
