import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { Architecture, AssertionFailure } from "./domain"

// Installer utilities can be PE32 even when the application target is x64.
// This is an inspected binary architecture, not another supported application target.
export const NativeImage = Schema.Union(
  Schema.Struct({ format: Schema.Literal("mach-o", "elf"), architectures: Schema.NonEmptyArray(Architecture) }),
  Schema.Struct({ format: Schema.Literal("pe"), architectures: Schema.NonEmptyArray(Schema.Literal("x86", "x64", "arm64")) }),
)
export type NativeImage = typeof NativeImage.Type
/** Architecture inspection only; execution and signature validation are separate acceptance checks. */
export const decodeNativeImage = (bytes: Uint8Array) => Effect.try({
  try: (): NativeImage => {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    const require = (condition: boolean, message: string) => { if (!condition) throw new Error(message) }
    require(bytes.length >= 32, "Native image header is truncated")
    const arch = (value: number, x64: number, arm64: number): typeof Architecture.Type => {
      require(value === x64 || value === arm64, "Native image uses an unsupported CPU architecture")
      return value === x64 ? "x64" : "arm64"
    }
    const magic = view.getUint32(0, false)
    if (magic === 0xcffaedfe || magic === 0xfeedfacf) {
      return { format: "mach-o", architectures: [arch(view.getUint32(4, magic === 0xcffaedfe), 0x01000007, 0x0100000c)] }
    }
    if ([0xcafebabe, 0xcafebabf, 0xbebafeca, 0xbfbafeca].includes(magic)) {
      const little = magic === 0xbebafeca || magic === 0xbfbafeca
      const stride = magic === 0xcafebabf || magic === 0xbfbafeca ? 32 : 20
      const count = view.getUint32(4, little)
      require(count > 0 && count <= 32 && bytes.length >= 8 + count * stride, "Universal image architecture table is invalid")
      const architectures: [typeof Architecture.Type, ...Array<typeof Architecture.Type>] = [arch(view.getUint32(8, little), 0x01000007, 0x0100000c)]
      for (let i = 1; i < count; i++) architectures.push(arch(view.getUint32(8 + i * stride, little), 0x01000007, 0x0100000c))
      return { format: "mach-o", architectures }
    }
    if (magic === 0x7f454c46) {
      require(bytes[4] === 2 && (bytes[5] === 1 || bytes[5] === 2) && bytes[6] === 1 && bytes.length >= 64, "Expected a valid ELF64 header")
      return { format: "elf", architectures: [arch(view.getUint16(18, bytes[5] === 1), 62, 183)] }
    }
    if (view.getUint16(0, true) === 0x5a4d) {
      require(bytes.length >= 64, "DOS header is truncated")
      const pe = view.getUint32(60, true)
      require(pe >= 64 && pe + 26 <= bytes.length, "PE header is outside the bounded header read")
      require(view.getUint32(pe, true) === 0x00004550, "Invalid PE signature")
      const machine = view.getUint16(pe + 4, true), optional = view.getUint16(pe + 24, true)
      if (machine === 0x14c) {
        require(optional === 0x10b, "x86 image must use PE32 headers")
        return { format: "pe", architectures: ["x86"] }
      }
      require(optional === 0x20b, "64-bit image must use PE32+ headers")
      return { format: "pe", architectures: [arch(machine, 0x8664, 0xaa64)] }
    }
    throw new Error("File has no supported native image header")
  },
  catch: error => new AssertionFailure({ message: error instanceof Error ? error.message : "Invalid native image" }),
})

export const inspectNativeImage = (path: string) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const file = yield* fs.open(path)
  const bytes = yield* file.readAlloc(64 * 1024)
  if (Option.isNone(bytes)) return yield* new AssertionFailure({ message: `Native image is empty: ${path}` })
  return yield* decodeNativeImage(bytes.value)
})).pipe(Effect.mapError(error => error._tag === "AssertionFailure" ? error : new AssertionFailure({ message: `Cannot inspect native package file: ${path}: ${error.message}` })))
