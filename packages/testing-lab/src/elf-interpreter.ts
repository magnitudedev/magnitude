import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { Architecture, AssertionFailure } from "./domain"
import { inspectNativeImage } from "./native-image"
import { checkedCommand } from "./process"

export const ElfInterpreter = Schema.Struct({ path: Schema.optionalWith(Schema.NonEmptyString, { as: "Option", exact: true }) })
const fail = (message: string) => new AssertionFailure({ message: `ELF interpreter: ${message}` })
export const decodeElfInterpreter = (output: string, arch: typeof Architecture.Type) => Effect.gen(function* () {
  if (!/^Elf file type is (?:DYN|EXEC)\b/m.test(output) || !/^Program Headers:\s*$/m.test(output)) return yield* fail("missing native program headers")
  const segments = [...output.matchAll(/^\s*INTERP\s/gm)]
  const paths = [...output.matchAll(/\[Requesting program interpreter:\s*([^\]]+)\]/g)].map(match => match[1]!.trim())
  if (segments.length !== paths.length || paths.length > 1) return yield* fail("missing or ambiguous interpreter segment")
  if (!paths.length) return ElfInterpreter.make({ path: Option.none() })
  const expected = arch === "x64" ? "/lib64/ld-linux-x86-64.so.2" : "/lib/ld-linux-aarch64.so.1"
  if (paths[0] !== expected) return yield* fail(`unexpected loader ${paths[0]}; expected ${expected}`)
  return ElfInterpreter.make({ path: Option.some(expected) })
})
export const inspectElfInterpreter = (file: string, arch: typeof Architecture.Type) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const report = yield* checkedCommand("readelf", ["--wide", "--program-headers", file], {
    env: { PATH: "/usr/sbin:/usr/bin:/sbin:/bin", LC_ALL: "C" }, inheritEnv: false, timeoutMs: 30_000, maxOutputBytes: 4 * 1024 * 1024,
  })
  const interpreter = yield* decodeElfInterpreter(report.stdout, arch)
  if (Option.isSome(interpreter.path)) {
    const path = yield* fs.realPath(interpreter.path.value)
    if (!/^\/(?:usr\/)?lib(?:64)?\//.test(path)) return yield* fail("system loader resolves outside OS library directories")
    const image = yield* inspectNativeImage(path)
    if (image.format !== "elf" || !image.architectures.includes(arch)) return yield* fail("system loader has the wrong native architecture")
  }
  return interpreter
})
