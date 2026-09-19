import { FileSystem } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { join, sep } from "node:path"
import { Architecture, AssertionFailure } from "./domain"
import { inspectNativeImage, NativeImage } from "./native-image"

export const NativeFile = Schema.Struct({ path: Schema.NonEmptyString, image: NativeImage })
const failure = (message: string) => new AssertionFailure({ message: `Native inventory: ${message}` })
const magicNumbers = new Set([0xcffaedfe, 0xfeedfacf, 0xcafebabe, 0xcafebabf, 0xbebafeca, 0xbfbafeca, 0x7f454c46])

/** Traverse package aliases once; never follow a link outside its installation. */
export const nativeInventory = (directory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.realPath(directory), pending = [root], visited = new Set<string>()
  const result: (typeof NativeFile.Type)[] = []
  for (let index = 0; index < pending.length; index++) {
    if (pending.length > 65_536) return yield* failure("package exceeds the file-count limit")
    const path = yield* fs.realPath(pending[index]!)
    if (path !== root && !path.startsWith(root + sep)) return yield* failure(`package link escapes its root: ${pending[index]}`)
    if (visited.has(path)) continue
    visited.add(path)
    const info = yield* fs.stat(path)
    if (info.type === "Directory") {
      for (const name of (yield* fs.readDirectory(path)).sort()) pending.push(join(path, name))
    } else if (info.type === "File") {
      const native = yield* Effect.scoped(Effect.gen(function* () {
        const file = yield* fs.open(path)
        const bytes = yield* file.readAlloc(4)
        if (Option.isNone(bytes) || bytes.value.length < 4) return false
        const view = new DataView(bytes.value.buffer, bytes.value.byteOffset, bytes.value.byteLength)
        return magicNumbers.has(view.getUint32(0, false)) || view.getUint16(0, true) === 0x5a4d
      }))
      if (native) result.push(NativeFile.make({ path, image: yield* inspectNativeImage(path) }))
    } else return yield* failure(`package contains a non-regular entry: ${path}`)
  }
  return result
})

/** Select the actual Mach-O slice and distinguish executables from libraries/bundles. */
export const machExecutable = (path: string, arch: typeof Architecture.Type) => Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const file = yield* fs.open(path)
  const header = yield* file.readAlloc(2048)
  if (Option.isNone(header) || header.value.length < 32) return yield* failure("truncated Mach-O header")
  let bytes = header.value
  let view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  let magic = view.getUint32(0, false)
  const cpu = arch === "arm64" ? 0x0100000c : 0x01000007
  if ([0xcafebabe, 0xcafebabf, 0xbebafeca, 0xbfbafeca].includes(magic)) {
    const little = magic === 0xbebafeca || magic === 0xbfbafeca
    const wide = magic === 0xcafebabf || magic === 0xbfbafeca, stride = wide ? 32 : 20
    const count = view.getUint32(4, little)
    if (count === 0 || count > 32 || 8 + count * stride > bytes.length) return yield* failure("invalid universal architecture table")
    const slices: { offset: bigint; size: bigint }[] = []
    for (let index = 0; index < count; index++) {
      const position = 8 + index * stride
      if (view.getUint32(position, little) === cpu) slices.push({
        offset: wide ? view.getBigUint64(position + 8, little) : BigInt(view.getUint32(position + 8, little)),
        size: wide ? view.getBigUint64(position + 16, little) : BigInt(view.getUint32(position + 12, little)),
      })
    }
    if (slices.length !== 1 || slices[0]!.offset < BigInt(8 + count * stride) || slices[0]!.size < 32n
      || slices[0]!.offset + slices[0]!.size > (yield* file.stat).size) return yield* failure("missing, ambiguous or truncated selected slice")
    yield* file.seek(slices[0]!.offset, "start")
    const slice = yield* file.readAlloc(32)
    if (Option.isNone(slice) || slice.value.length < 32) return yield* failure("truncated selected slice")
    bytes = slice.value
    view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    magic = view.getUint32(0, false)
  }
  if (magic !== 0xcffaedfe && magic !== 0xfeedfacf) return yield* failure("expected a Mach-O slice")
  const little = magic === 0xcffaedfe
  if (view.getUint32(4, little) !== cpu) return yield* failure("slice architecture differs from selected target")
  const kind = view.getUint32(12, little)
  if (![2, 6, 8].includes(kind)) return yield* failure(`unsupported packaged Mach-O file type ${kind}`)
  return kind === 2
}))
