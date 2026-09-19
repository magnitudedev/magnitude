import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { verifyElfDependencyClosure } from "../src/elf-dependency-closure"
import { inspectElfInterpreter } from "../src/elf-interpreter"
import { checkedCommand, command, ProcessExecutorLive } from "../src/process"
import { systemElfResolver } from "../src/system-elf"

test.runIf(process.platform === "linux")("native ELF inspection agrees with the loader on transitive libraries and incompatible symbol versions", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "lab-elf-native-" })
  const root = join(temporary, "candidate"), libraries = join(root, "libraries")
  yield* fs.makeDirectory(libraries, { recursive: true })
  const first = join(libraries, "libfirst.so"), second = join(libraries, "libsecond.so"), app = join(root, "app")
  yield* fs.writeFileString(join(temporary, "second.c"), "int second(void) { return 0; }\n")
  yield* fs.writeFileString(join(temporary, "first.c"), "int second(void); int first(void) { return second(); }\n")
  yield* fs.writeFileString(join(temporary, "main.c"), "int first(void); int main(void) { return first(); }\n")
  const versionScript = join(temporary, "versions.map")
  const compileSecond = (version: string) => Effect.gen(function* () {
    yield* fs.writeFileString(versionScript, `${version} { global: second; local: *; };\n`)
    yield* checkedCommand("cc", ["-shared", "-fPIC", join(temporary, "second.c"), "-Wl,-soname,libsecond.so", `-Wl,--version-script=${versionScript}`, "-o", second])
  })
  yield* compileSecond("SECOND_1.0")
  yield* checkedCommand("cc", ["-shared", "-fPIC", join(temporary, "first.c"), "-L", libraries, "-lsecond", "-Wl,-soname,libfirst.so", "-o", first])
  const compileApp = (tag: "rpath" | "runpath") => checkedCommand("cc", [join(temporary, "main.c"), "-L", libraries, "-lfirst",
    `-Wl,-rpath-link,${libraries}`, "-Wl,-rpath,$ORIGIN/libraries", tag === "rpath" ? "-Wl,--disable-new-dtags" : "-Wl,--enable-new-dtags", "-o", app])
  yield* compileApp("rpath")
  const arch = process.arch === "arm64" ? "arm64" : "x64"
  const manager = (yield* fs.exists("/usr/bin/dpkg-query")) ? "deb" : "rpm"
  const inspect = verifyElfDependencyClosure({ root, roots: [app], arch }).pipe(
    Effect.provide(systemElfResolver(manager, ["libc.so.6"])))
  const execute = command(app, [], { inheritEnv: false, env: { PATH: "/usr/bin:/bin", LC_ALL: "C" } })
  expect(Option.isSome((yield* inspectElfInterpreter(app, arch)).path)).toBe(true)
  expect((yield* execute).exitCode).toBe(0)
  expect((yield* inspect).files).toHaveLength(3)

  // Replacing a same-named library with an incompatible ABI must fail both checks.
  yield* compileSecond("SECOND_2.0")
  const incompatible = yield* inspect.pipe(Effect.either)
  expect(incompatible._tag).toBe("Left")
  if (incompatible._tag === "Left") expect(incompatible.left.message).toContain("SECOND_1.0")
  expect((yield* execute).exitCode).not.toBe(0)
  yield* compileSecond("SECOND_1.0")

  // RUNPATH is direct-only; the grandchild cannot inherit the executable's path.
  yield* compileApp("runpath")
  expect((yield* inspect.pipe(Effect.either))._tag).toBe("Left")
  expect((yield* execute).exitCode).not.toBe(0)
  yield* compileApp("rpath")
  expect((yield* inspect).files).toHaveLength(3)
  yield* fs.remove(second)
  expect((yield* inspect.pipe(Effect.either))._tag).toBe("Left")
  expect((yield* execute).exitCode).not.toBe(0)
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))), 60_000)
