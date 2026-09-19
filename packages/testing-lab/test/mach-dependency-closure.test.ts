import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { verifyMachDependencyClosure } from "../src/mach-dependency-closure"
import { checkedCommand, ProcessExecutorLive } from "../src/process"

test.runIf(process.platform === "darwin")("resolves actual transitive libraries and rejects missing, escaped and developer dependencies", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "lab-macho-closure-" })
  const root = join(temporary, "candidate"), libs = join(root, "libraries")
  yield* fs.makeDirectory(libs, { recursive: true })
  const first = join(libs, "libfirst.dylib"), second = join(libs, "libsecond.dylib"), executable = join(root, "app")
  yield* fs.writeFileString(join(temporary, "second.c"), "int second(void) { return 0; }\n")
  yield* fs.writeFileString(join(temporary, "first.c"), "int second(void); int first(void) { return second(); }\n")
  yield* fs.writeFileString(join(temporary, "main.c"), "int first(void); int main(void) { return first(); }\n")
  yield* checkedCommand("/usr/bin/clang", ["-dynamiclib", join(temporary, "second.c"), "-Wl,-install_name,@rpath/libsecond.dylib", "-o", second])
  yield* checkedCommand("/usr/bin/clang", ["-dynamiclib", join(temporary, "first.c"), second, "-Wl,-install_name,@rpath/libfirst.dylib", "-Wl,-rpath,@loader_path", "-o", first])
  yield* checkedCommand("/usr/bin/clang", [join(temporary, "main.c"), first, "-Wl,-rpath,@executable_path/libraries", "-o", executable])
  const run = verifyMachDependencyClosure({ root, executable, roots: [executable], arch: process.arch === "arm64" ? "arm64" : "x64" })
  const result = yield* run
  expect(result.files).toHaveLength(3)
  expect(result.edges.filter(edge => edge.resolution.kind === "owned").map(edge => edge.name)).toEqual(["@rpath/libfirst.dylib", "@rpath/libsecond.dylib"])
  expect(result.edges.some(edge => edge.resolution.kind === "system" && edge.name === "/usr/lib/libSystem.B.dylib")).toBe(true)
  yield* checkedCommand("/usr/bin/install_name_tool", ["-delete_rpath", "@loader_path", first])
  // The intermediate library now relies entirely on the executable's inherited path.
  expect((yield* run).files).toHaveLength(3)
  yield* fs.rename(second, join(temporary, "outside.dylib"))
  const absent = yield* run.pipe(Effect.either)
  expect(absent._tag).toBe("Left")
  if (absent._tag === "Left") expect(absent.left.message).toContain("unresolved required import @rpath/libsecond.dylib")
  yield* fs.symlink(join(temporary, "outside.dylib"), second)
  const escaped = yield* run.pipe(Effect.either)
  expect(escaped._tag).toBe("Left")
  if (escaped._tag === "Left") expect(escaped.left.message).toContain("not an owned regular file")
  yield* fs.remove(second)
  yield* fs.rename(join(temporary, "outside.dylib"), second)
  yield* checkedCommand("/usr/bin/install_name_tool", ["-add_rpath", "/opt/homebrew/lib", executable])
  const ambient = yield* run.pipe(Effect.either)
  expect(ambient._tag).toBe("Left")
  if (ambient._tag === "Left") expect(ambient.left.message).toContain("unowned or relative loader path")
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
