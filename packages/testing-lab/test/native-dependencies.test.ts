import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { decodeElfDependencies, decodeMachDependencies, decodePeDependencies, inspectNativeDependencies, NativeDependencies } from "../src/native-dependencies"
import { AssertionFailure } from "../src/domain"
import { checkedCommand, ProcessExecutorLive } from "../src/process"

test("Mach-O imports preserve spaces and linkage without treating install IDs as imports", async () => {
  const result = await Effect.runPromise(decodeMachDependencies(`candidate:
Load command 0
 cmd LC_ID_DYLIB
 name @rpath/This Library.dylib (offset 24)
Load command 1
 cmd LC_RPATH
 path @loader_path/../Private Libraries (offset 12)
Load command 2
 cmd LC_LOAD_DYLIB
 name @rpath/Electron Framework.framework/Electron Framework (offset 24)
Load command 3
 cmd LC_LOAD_WEAK_DYLIB
 name /System/Library/Frameworks/Optional.framework/Optional (offset 24)
Load command 4
 cmd LC_REEXPORT_DYLIB
 name @loader_path/owned.dylib (offset 24)
Load command 5
 cmd LC_LAZY_LOAD_DYLIB
 name @rpath/delayed.dylib (offset 24)
`))
  expect(result.rpaths).toEqual(["@loader_path/../Private Libraries"])
  expect(result.imports.map(value => [value.name, value.linkage])).toEqual([
    ["@rpath/Electron Framework.framework/Electron Framework", "required"],
    ["/System/Library/Frameworks/Optional.framework/Optional", "weak"],
    ["@loader_path/owned.dylib", "required"], ["@rpath/delayed.dylib", "delay"],
  ])
})

test("ELF keeps RPATH and RUNPATH distinct, including unsafe empty search components", async () => {
  const result = await Effect.runPromise(decodeElfDependencies(`Dynamic section at offset 0x123 contains 5 entries:
 Tag Type Name/Value
 0x1 (NEEDED) Shared library: [libowned.so]
 0x1 (NEEDED) Shared library: [/developer/libabsolute.so]
 0xf (RPATH) Library rpath: [$ORIGIN:]
 0x1d (RUNPATH) Library runpath: [$ORIGIN/../runtime:/usr/local/lib]
 0x0 (NULL) 0x0
`))
  expect(result.imports.map(value => value.name)).toEqual(["libowned.so", "/developer/libabsolute.so"])
  expect(result.rpaths).toEqual(["$ORIGIN", ""])
  expect(result.runpaths).toEqual(["$ORIGIN/../runtime", "/usr/local/lib"])
  expect((await Effect.runPromise(decodeElfDependencies("There is no dynamic section in this file.\n"))).imports).toEqual([])
})

test("PE preserves required and delay imports without importing the inspected file itself", async () => {
  const result = await Effect.runPromise(decodePeDependencies(`Dump of file C:\\build\\owned.dll
File Type: DLL
 Image has the following dependencies:
    KERNEL32.dll
    VCRUNTIME140.dll
 Image has the following delay load dependencies:
    nvcuda.dll
 Summary
    1000 .data
`))
  expect(result.imports).toEqual([{ name: "kernel32.dll", linkage: "required" },
    { name: "vcruntime140.dll", linkage: "required" }, { name: "nvcuda.dll", linkage: "delay" }])
})

test("missing or malformed reports cannot become empty successful dependency graphs", async () => {
  const cases: ReadonlyArray<Effect.Effect<NativeDependencies, AssertionFailure>> = [decodeMachDependencies("not a Mach-O file"), decodeMachDependencies("Load command 0\n cmd LC_LOAD_DYLIB\n"),
    decodeMachDependencies("Load command 0\n cmd LC_RPATH\n"), decodeElfDependencies(""),
    decodeElfDependencies("Dynamic section at offset 0x1 contains 2 entries:\n 0x1 (NEEDED) missing\n 0x0 (NULL) 0x0\n"),
    decodeElfDependencies("Dynamic section at offset 0x1 contains 1 entries:\n"),
    decodePeDependencies("Dump of file library.dll"), decodePeDependencies("File Type: DLL\nImage has the following dependencies:\n C:\\ambient\\bad.dll\nSummary\n")]
  for (const result of cases) expect((await Effect.runPromise(result.pipe(Effect.either)))._tag).toBe("Left")
})

test.runIf(process.platform === "darwin")("reads actual native import declarations after the linked library is removed", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-native-dependencies-" })
  const library = join(root, "libowned.dylib"), executable = join(root, "candidate (GPU)")
  yield* fs.writeFileString(join(root, "library.c"), "int owned(void) { return 0; }\n")
  yield* fs.writeFileString(join(root, "main.c"), "int owned(void); int main(void) { return owned(); }\n")
  yield* checkedCommand("/usr/bin/clang", ["-dynamiclib", join(root, "library.c"), "-Wl,-install_name,@rpath/libowned.dylib", "-o", library])
  yield* checkedCommand("/usr/bin/clang", [join(root, "main.c"), library, "-Wl,-rpath,@executable_path", "-o", executable])
  yield* fs.remove(library)
  const report = yield* inspectNativeDependencies(executable, process.arch === "arm64" ? "arm64" : "x64")
  expect(report.format).toBe("mach-o")
  expect(report.imports).toContainEqual({ name: "@rpath/libowned.dylib", linkage: "required" })
  if (report.format === "mach-o") expect(report.rpaths).toEqual(["@executable_path"])
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
