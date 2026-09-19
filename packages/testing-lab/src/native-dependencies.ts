import { Effect, Schema } from "effect"
import { FileSystem } from "@effect/platform"
import { join, resolve } from "node:path"
import { Architecture, AssertionFailure } from "./domain"
import { inspectNativeImage } from "./native-image"
import { checkedCommand } from "./process"

const Import = Schema.Struct({ name: Schema.NonEmptyString, linkage: Schema.Literal("required", "weak", "delay") })
export const NativeDependencies = Schema.Union(
  Schema.Struct({ format: Schema.Literal("mach-o"), imports: Schema.Array(Import), rpaths: Schema.Array(Schema.String) }),
  Schema.Struct({ format: Schema.Literal("elf"), imports: Schema.Array(Import), rpaths: Schema.Array(Schema.String), runpaths: Schema.Array(Schema.String) }),
  Schema.Struct({ format: Schema.Literal("pe"), imports: Schema.Array(Import) }),
)
export type NativeDependencies = typeof NativeDependencies.Type
const invalid = (message: string) => new AssertionFailure({ message: `Native dependency report: ${message}` })

/** Read declarations only. Resolving the complete owned graph is a separate acceptance step. */
export const decodeMachDependencies = (report: string) => Effect.gen(function* () {
  const blocks = report.split(/^Load command \d+\s*$/m).slice(1)
  if (blocks.length === 0) return yield* invalid("missing Mach-O load commands")
  const imports: (typeof Import.Type)[] = [], rpaths: string[] = []
  for (const block of blocks) {
    const command = /^\s*cmd (LC_[A-Z0-9_]+)\s*$/m.exec(block)?.[1]
    if (!command) return yield* invalid("load command has no type")
    if (command === "LC_RPATH") {
      const path = /^\s*path (.+) \(offset \d+\)\s*$/m.exec(block)?.[1]
      if (!path) return yield* invalid("RPATH has no path")
      rpaths.push(path)
    } else if (["LC_LOAD_DYLIB", "LC_LOAD_WEAK_DYLIB", "LC_REEXPORT_DYLIB", "LC_LOAD_UPWARD_DYLIB", "LC_LAZY_LOAD_DYLIB"].includes(command)) {
      const name = /^\s*name (.+) \(offset \d+\)\s*$/m.exec(block)?.[1]
      if (!name) return yield* invalid("library command has no name")
      imports.push({ name, linkage: command === "LC_LOAD_WEAK_DYLIB" ? "weak" : command === "LC_LAZY_LOAD_DYLIB" ? "delay" : "required" })
    }
    // LC_ID_DYLIB identifies this library; it is not an edge to another library.
  }
  return { format: "mach-o" as const, imports, rpaths }
})

export const decodeElfDependencies = (report: string) => Effect.gen(function* () {
  if (report.trim() === "There is no dynamic section in this file.") {
    return { format: "elf" as const, imports: [], rpaths: [], runpaths: [] }
  }
  if (!/^Dynamic section at offset .* contains \d+ entries:\s*$/m.test(report)) return yield* invalid("missing ELF dynamic section")
  if (!/^\s*0x[0-9a-f]+\s+\(NULL\)\s+/m.test(report)) return yield* invalid("incomplete ELF dynamic section")
  const imports: (typeof Import.Type)[] = [], rpaths: string[] = [], runpaths: string[] = []
  for (const line of report.split("\n")) {
    const tag = /\((NEEDED|RPATH|RUNPATH)\)/.exec(line)?.[1]
    if (!tag) continue
    const value = /\[([^\]]*)\]\s*$/.exec(line)?.[1]
    if (value === undefined || (tag === "NEEDED" && value.length === 0)) return yield* invalid(`malformed ELF ${tag}`)
    if (tag === "NEEDED") imports.push({ name: value, linkage: "required" })
    else (tag === "RPATH" ? rpaths : runpaths).push(...value.split(":"))
  }
  // Empty path components are retained: they mean the working directory, not no search path.
  return { format: "elf" as const, imports, rpaths, runpaths }
})

export const decodePeDependencies = (report: string) => Effect.gen(function* () {
  if (!/^\s*File Type: (?:EXECUTABLE IMAGE|DLL)\s*$/m.test(report) || !/^\s*Summary\s*$/m.test(report)) {
    return yield* invalid("missing PE image header or completed dependency report")
  }
  let linkage: "required" | "delay" | undefined
  const imports: (typeof Import.Type)[] = []
  for (const line of report.split("\n")) {
    if (/^\s*Image has the following dependencies:\s*$/.test(line)) linkage = "required"
    else if (/^\s*Image has the following delay load dependencies:\s*$/.test(line)) linkage = "delay"
    else if (/^\s*Summary\s*$/.test(line)) linkage = undefined
    else if (linkage && line.trim()) {
      const name = /^\s+([\w.-]+\.dll)\s*$/i.exec(line)?.[1]
      if (!name) return yield* invalid("malformed PE dependency name")
      imports.push({ name: name.toLowerCase(), linkage })
    }
  }
  return { format: "pe" as const, imports }
})

/** Inspect the selected native slice without executing candidate code or consulting ambient loader overrides. */
export const inspectNativeDependencies = (path: string, arch: typeof Architecture.Type) => Effect.gen(function* () {
  const image = yield* inspectNativeImage(path)
  if (!image.architectures.includes(arch)) return yield* invalid("requested architecture is absent")
  const environment = { PATH: process.env.PATH ?? "", LC_ALL: "C", ...(process.env.SystemRoot ? { SystemRoot: process.env.SystemRoot } : {}) }
  const options = { env: environment, inheritEnv: false, timeoutMs: 30_000, maxOutputBytes: 4 * 1024 * 1024 }
  if (image.format === "mach-o") {
    return yield* Effect.scoped(Effect.gen(function* () {
      // otool treats trailing parentheses as archive-member syntax, even in an argv
      // filename. An owned temporary alias changes only inspection, never loader context.
      const fs = yield* FileSystem.FileSystem
      const temporary = yield* fs.makeTempDirectoryScoped({ directory: "/tmp", prefix: "ml-mach-inspect-" })
      const alias = join(temporary, "image")
      yield* fs.symlink(resolve(path), alias)
      const output = yield* checkedCommand("/usr/bin/otool", ["-arch", arch === "arm64" ? "arm64" : "x86_64", "-l", alias], options)
      return yield* decodeMachDependencies(output.stdout)
    }))
  }
  if (image.format === "elf") {
    const output = yield* checkedCommand("readelf", ["--wide", "--dynamic", path], options)
    return yield* decodeElfDependencies(output.stdout)
  }
  const output = yield* checkedCommand("dumpbin.exe", ["/nologo", "/dependents", path], options)
  return yield* decodePeDependencies(output.stdout)
})
