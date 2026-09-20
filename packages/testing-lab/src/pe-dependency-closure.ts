import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { win32 } from "node:path"
import { AssertionFailure } from "./domain"
import { inspectNativeImage, type NativeImage } from "./native-image"
import { inspectPeGraph, pePathKey, verifyPeImports } from "./pe-dependencies"
import { verifyWindowsSignature } from "./suites/windows-package-trust"

const Resolution = Schema.Union(
  Schema.Struct({ kind: Schema.Literal("owned"), path: Schema.String }),
  Schema.Struct({ kind: Schema.Literal("system"), path: Schema.String, publisher: Schema.String }),
)
export const PeDependencyClosure = Schema.Struct({ root: Schema.String, files: Schema.Array(Schema.String),
  edges: Schema.Array(Schema.Struct({ executable: Schema.String, from: Schema.String, name: Schema.String,
    linkage: Schema.Literal("required", "delay"), resolution: Resolution })) })
const fail = (message: string) => new AssertionFailure({ message: `PE dependency closure: ${message}` })

/** Inspect each packaged native root, preserving that root's native loader search context. */
export const verifyPeDependencyClosure = (input: { readonly root: string; readonly roots: readonly string[];
  readonly inspector: string; readonly systemRoot: string; readonly cuda: boolean; readonly ownedSearchPaths: readonly string[] }) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.realPath(input.root), rootKey = pePathKey(root)
  const systemRoot = yield* fs.realPath(input.systemRoot)
  const workspace = yield* fs.makeTempDirectoryScoped({ prefix: "lab-pe-inspection-" })
  const inside = (path: string, directory: string) => pePathKey(path).startsWith(pePathKey(directory) + "\\")
  const environment = { SystemRoot: systemRoot, WINDIR: systemRoot,
    PATH: `${systemRoot}\\System32\\WindowsPowerShell\\v1.0;${systemRoot}\\System32;${systemRoot}`, TEMP: workspace, TMP: workspace }
  const ownedSearchPaths: string[] = []
  for (const directory of input.ownedSearchPaths) {
    const canonical = yield* fs.realPath(directory)
    if (!inside(canonical, root) || (yield* fs.stat(canonical)).type !== "Directory") return yield* fail(`loader search path escapes the installation: ${directory}`)
    ownedSearchPaths.push(canonical)
  }
  const images = new Map<string, Extract<NativeImage, { format: "pe" }>>()
  const system = new Map<string, { readonly path: string; readonly publisher: string }>()
  const files = new Set<string>(), edges: (typeof PeDependencyClosure.Type)["edges"][number][] = []
  if (!input.roots.length || !win32.isAbsolute(input.inspector)) return yield* fail("missing native roots or absolute inspector path")
  const image = (file: string) => Effect.gen(function* () {
    const key = pePathKey(file), existing = images.get(key)
    if (existing) return existing
    const result = yield* inspectNativeImage(file)
    if (result.format !== "pe" || result.architectures.length !== 1) return yield* fail(`not a single-architecture PE image: ${file}`)
    images.set(key, result)
    return result
  })
  const osBoundary = (file: string, name: string, arch: string) => Effect.gen(function* () {
    const expected = win32.join(systemRoot, arch === "x86" ? "SysWOW64" : "System32")
    if (!inside(file, expected) && !inside(file, win32.join(systemRoot, "WinSxS"))) return yield* fail(`unowned loader resolution: ${file}`)
    const observed = yield* image(file)
    if (!observed.architectures.some(value => value === arch)) return yield* fail(`wrong system dependency architecture: ${file}`)
    const key = pePathKey(file), cached = system.get(key)
    if (cached && (cached.publisher !== "NVIDIA Corporation" || (input.cuda && name === "nvcuda.dll"))) return cached
    const signature = yield* verifyWindowsSignature(file, { kind: "development" }, environment)
    if (signature.status !== "Valid" || Option.isNone(signature.publisher)) return yield* fail(`system dependency has no valid publisher: ${file}`)
    const publisher = signature.publisher.value
    const microsoft = ["Microsoft Windows", "Microsoft Corporation", "Microsoft Windows Publisher", "Microsoft Windows Software Compatibility Publisher"]
    if (!microsoft.includes(publisher) && !(input.cuda && name === "nvcuda.dll" && publisher === "NVIDIA Corporation")) {
      return yield* fail(`unexpected system dependency publisher ${publisher}: ${file}`)
    }
    const result = { path: file, publisher }
    system.set(key, result)
    return result
  })
  for (const entrypoint of input.roots) {
    const executable = yield* fs.realPath(entrypoint)
    if (!inside(executable, root) || pePathKey(executable) === rootKey) return yield* fail(`native root escapes the installation: ${entrypoint}`)
    const executableImage = yield* image(executable), architecture = executableImage.architectures[0]!
    const graph = yield* inspectPeGraph(input.inspector, executable, root, workspace, systemRoot, ownedSearchPaths)
    const pending = [executable], visited = new Set<string>()
    for (let index = 0; index < pending.length; index++) {
      if (pending.length > 16_384) return yield* fail("owned graph exceeds inspection limit")
      const file = pending[index]!, key = pePathKey(file)
      if (visited.has(key)) continue
      visited.add(key); files.add(file)
      if (!(yield* image(file)).architectures.includes(architecture)) return yield* fail(`mixed architecture in ${executable}: ${file}`)
      const module = graph.get(key)
      if (!module) return yield* fail(`loader report omitted owned file: ${file}`)
      const required = yield* verifyPeImports(module.Imports)
      for (const declaration of required) {
        const matches = module.Dependencies.filter(item => item.ModuleName.toLowerCase() === declaration.name)
        if (!matches.length || matches.some(match => match.Filepath === null)
          || new Set(matches.map(match => pePathKey(match.Filepath!))).size !== 1) return yield* fail(`unresolved or ambiguous ${declaration.name} from ${file}`)
        const selected = yield* fs.realPath(matches[0]!.Filepath!)
        if ((yield* fs.stat(selected)).type !== "File") return yield* fail(`dependency is not a file: ${selected}`)
        const resolution = inside(selected, root)
          ? { kind: "owned" as const, path: selected }
          : { kind: "system" as const, ...(yield* osBoundary(selected, declaration.name, architecture)) }
        edges.push({ executable, from: file, ...declaration, resolution })
        if (resolution.kind === "owned") pending.push(selected)
      }
      if (module.Dependencies.some(item => !required.some(declaration => declaration.name === item.ModuleName.toLowerCase()))) {
        return yield* fail(`loader and import declarations disagree for ${file}`)
      }
    }
  }
  return PeDependencyClosure.make({ root, files: [...files].sort(), edges })
}))
