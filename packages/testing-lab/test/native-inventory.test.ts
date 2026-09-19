import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { machExecutable, nativeInventory } from "../src/native-inventory"

const thin = (arch: "arm64" | "x64", kind: number) => {
  const bytes = new Uint8Array(64), view = new DataView(bytes.buffer)
  view.setUint32(0, 0xfeedfacf, true)
  view.setUint32(4, arch === "arm64" ? 0x0100000c : 0x01000007, true)
  view.setUint32(12, kind, true)
  return bytes
}

test("inventory deduplicates framework aliases, rejects escaping links and does not inspect text as native", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "lab-native-inventory-" })
  const root = join(temporary, "bundle"), native = join(root, "Versions", "A", "native")
  yield* fs.makeDirectory(join(root, "Versions", "A"), { recursive: true })
  yield* fs.writeFile(native, thin("arm64", 6))
  yield* fs.writeFileString(join(root, "readme"), "ordinary text")
  yield* fs.symlink("A", join(root, "Versions", "Current"))
  yield* fs.symlink("Versions/Current/native", join(root, "native"))
  expect((yield* nativeInventory(root)).map(file => file.path)).toEqual([yield* fs.realPath(native)])
  yield* fs.writeFile(join(temporary, "outside"), thin("arm64", 6))
  yield* fs.symlink(join(temporary, "outside"), join(root, "escape"))
  expect((yield* nativeInventory(root).pipe(Effect.either))._tag).toBe("Left")
})).pipe(Effect.provide(BunContext.layer))))

test("Mach-O role inspection reads the selected universal slice and rejects ambiguous or invalid slices", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-mach-role-" })
  const path = join(root, "native")
  for (const wide of [false, true]) for (const little of [false, true]) {
    const bytes = new Uint8Array(512), view = new DataView(bytes.buffer), stride = wide ? 32 : 20
    view.setUint32(0, wide ? 0xcafebabf : 0xcafebabe, little)
    view.setUint32(4, 2, little)
    for (const [index, cpu] of [0x01000007, 0x0100000c].entries()) {
      const position = 8 + index * stride, offset = 256 + 64 * index
      view.setUint32(position, cpu, little)
      if (wide) { view.setBigUint64(position + 8, BigInt(offset), little); view.setBigUint64(position + 16, 64n, little) }
      else { view.setUint32(position + 8, offset, little); view.setUint32(position + 12, 64, little) }
      bytes.set(thin(index === 0 ? "x64" : "arm64", index === 0 ? 2 : 6), offset)
    }
    yield* fs.writeFile(path, bytes)
    expect(yield* machExecutable(path, "x64")).toBe(true)
    expect(yield* machExecutable(path, "arm64")).toBe(false)
    view.setUint32(8 + stride, 0x01000007, little)
    yield* fs.writeFile(path, bytes)
    expect((yield* machExecutable(path, "x64").pipe(Effect.either))._tag).toBe("Left")
  }
  for (const kind of [1, 9]) {
    yield* fs.writeFile(path, thin("arm64", kind))
    expect((yield* machExecutable(path, "arm64").pipe(Effect.either))._tag).toBe("Left")
  }
  yield* fs.writeFile(path, thin("arm64", 8))
  expect(yield* machExecutable(path, "arm64")).toBe(false)
  expect((yield* machExecutable(path, "x64").pipe(Effect.either))._tag).toBe("Left")
})).pipe(Effect.provide(BunContext.layer))))
