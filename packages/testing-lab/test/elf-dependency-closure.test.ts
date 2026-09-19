import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { AssertionFailure } from "../src/domain"
import { SystemElfResolver, verifyElfDependencyClosure } from "../src/elf-dependency-closure"
import { ProcessExecutor } from "../src/process"

test("ELF graph requires transitive owned libraries and records verified system boundaries", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temp = yield* fs.makeTempDirectoryScoped({ prefix: "lab-elf-graph-" })
  const root = yield* fs.realPath(temp)
  yield* fs.makeDirectory(join(root, "lib"))
  const executable = join(root, "app"), first = join(root, "lib", "first.so"), second = join(root, "lib", "second.so")
  const bytes = new Uint8Array(64), view = new DataView(bytes.buffer)
  bytes.set([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1]); view.setUint16(18, 62, true)
  for (const file of [executable, first, second]) yield* fs.writeFile(file, bytes)
  let modern = false, badPath = false, abi: "none" | "matching" | "missing" = "none"
  const seen: string[] = []
  const graph = verifyElfDependencyClosure({ root, roots: [executable], arch: "x64" }).pipe(Effect.provide([
    Layer.succeed(ProcessExecutor, { run: spec => {
      const file = spec.args[spec.args.length - 1]!
      if (spec.args.includes("--version-info")) {
        const stdout = abi !== "none" && file === executable
          ? "Version needs section '.gnu.version_r' contains 1 entry:\n 000000: Version: 1 File: first.so Cnt: 1\n 0x0010: Name: FIRST_2.0 Flags: none Version: 2\n"
          : abi === "matching" && file === first
          ? "Version definition section '.gnu.version_d' contains 1 entry:\n 000000: Rev: 1 Flags: none Index: 2 Cnt: 1 Name: FIRST_2.0\n"
          : "No version information found in this file.\n"
        return Effect.succeed({ exitCode: 0, stderr: "", stdout })
      }
      const imports = file === executable ? [badPath ? "/developer/first.so" : "first.so"] : file === first ? ["second.so"] : ["libc.so.6"]
      const paths = file === executable ? ` 0x1 (${modern ? "RUNPATH" : "RPATH"}) Library path: [$ORIGIN/lib]\n` : ""
      return Effect.succeed({ exitCode: 0, stderr: "", stdout: `Dynamic section at offset 0x100 contains 3 entries:\n${imports.map(name => ` 0x1 (NEEDED) Shared library: [${name}]\n`).join("")}${paths} 0x0 (NULL) 0x0\n` })
    } }),
    Layer.succeed(SystemElfResolver, { resolve: name => {
      seen.push(name)
      return name === "libc.so.6" ? Effect.succeed({ path: "/usr/lib/x86_64-linux-gnu/libc.so.6", package: "libc6:amd64" })
        : Effect.fail(new AssertionFailure({ message: `No OS package supplies ${name}` }))
    } }),
  ]))
  const result = yield* graph
  expect(result.files).toHaveLength(3)
  expect(seen).toEqual(["libc.so.6"])
  expect(result.edges.at(-1)?.resolution).toEqual({ kind: "system", library: { path: "/usr/lib/x86_64-linux-gnu/libc.so.6", package: "libc6:amd64" } })
  abi = "matching"
  expect((yield* graph).files).toHaveLength(3)
  abi = "missing"
  const incompatible = yield* graph.pipe(Effect.either)
  expect(incompatible._tag).toBe("Left")
  if (incompatible._tag === "Left") expect(incompatible.left.message).toContain("does not define required versions: FIRST_2.0")
  abi = "none"
  modern = true
  const nonInherited = yield* graph.pipe(Effect.either)
  expect(nonInherited._tag).toBe("Left")
  if (nonInherited._tag === "Left") expect(nonInherited.left.message).toContain("second.so")
  modern = false
  yield* fs.remove(second)
  expect((yield* graph.pipe(Effect.either))._tag).toBe("Left")
  badPath = true
  const ambient = yield* graph.pipe(Effect.either)
  expect(ambient._tag).toBe("Left")
  if (ambient._tag === "Left") expect(ambient.left.message).toContain("unowned DT_NEEDED")
})).pipe(Effect.provide(BunContext.layer))))
