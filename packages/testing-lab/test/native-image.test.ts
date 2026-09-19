import { expect, test } from "vitest"
import { Effect } from "effect"
import { decodeNativeImage } from "../src/native-image"

const header = (format: "mach-o" | "elf" | "pe", arch: "x64" | "arm64") => {
  const bytes = new Uint8Array(256), view = new DataView(bytes.buffer)
  if (format === "mach-o") { view.setUint32(0, 0xfeedfacf, true); view.setUint32(4, arch === "x64" ? 0x01000007 : 0x0100000c, true) }
  if (format === "elf") { bytes.set([0x7f, 0x45, 0x4c, 0x46, 2, 1, 1]); view.setUint16(18, arch === "x64" ? 62 : 183, true) }
  if (format === "pe") { view.setUint16(0, 0x5a4d, true); view.setUint32(60, 128, true); view.setUint32(128, 0x00004550, true); view.setUint16(132, arch === "x64" ? 0x8664 : 0xaa64, true); view.setUint16(152, 0x20b, true) }
  return bytes
}
for (const format of ["mach-o", "elf", "pe"] as const) for (const arch of ["x64", "arm64"] as const) test(`reads ${format} ${arch} from header bytes`, async () => {
  expect(await Effect.runPromise(decodeNativeImage(header(format, arch)))).toEqual({ format, architectures: [arch] })
})
test("reads universal Mach-O architecture tables in both formats and byte orders", async () => {
  for (const wide of [false, true]) for (const little of [false, true]) {
    const bytes = new Uint8Array(128), view = new DataView(bytes.buffer), stride = wide ? 32 : 20
    view.setUint32(0, wide ? 0xcafebabf : 0xcafebabe, little); view.setUint32(4, 2, little)
    view.setUint32(8, 0x01000007, little); view.setUint32(8 + stride, 0x0100000c, little)
    expect((await Effect.runPromise(decodeNativeImage(bytes))).architectures).toEqual(["x64", "arm64"])
  }
})
test("rejects scripts, truncation, unsupported machines and malformed offsets", async () => {
  const malformed = [new Uint8Array(), new TextEncoder().encode("#!/bin/sh\necho arm64 version 0.1.3\n")]
  const pe = header("pe", "x64"); new DataView(pe.buffer).setUint32(60, 0xfffffff0, true); malformed.push(pe)
  const elf = header("elf", "arm64"); elf[4] = 1; malformed.push(elf)
  const machine = header("mach-o", "x64"); new DataView(machine.buffer).setUint32(4, 7, true); malformed.push(machine)
  const fat = new Uint8Array(64); new DataView(fat.buffer).setUint32(0, 0xcafebabe); new DataView(fat.buffer).setUint32(4, 100); malformed.push(fat)
  for (const bytes of malformed) expect((await Effect.runPromise(decodeNativeImage(bytes).pipe(Effect.either)))._tag).toBe("Left")
})
