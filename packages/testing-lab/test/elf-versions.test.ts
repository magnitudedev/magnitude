import { Effect } from "effect"
import { expect, test } from "vitest"
import { attestElfVersions, decodeElfVersions } from "../src/elf-versions"

const report = `Version definition section '.gnu.version_d' contains 2 entries:
 Addr: 0x0000000000001000 Offset: 0x001000 Link: 4 (.dynstr)
 000000: Rev: 1 Flags: BASE Index: 1 Cnt: 1 Name: libexample.so
 0x001c: Rev: 1 Flags: none Index: 2 Cnt: 1 Name: EXAMPLE_1.0
Version needs section '.gnu.version_r' contains 1 entry:
 Addr: 0x0000000000001020 Offset: 0x001020 Link: 4 (.dynstr)
 000000: Version: 1 File: libc.so.6 Cnt: 2
 0x0010: Name: GLIBC_2.34 Flags: none Version: 3
 0x0020: Name: GLIBC_2.2.5 Flags: none Version: 4
`
test("ELF ABI requirements bind exact version names to their providing library", () => Effect.runPromise(Effect.gen(function* () {
  const value = yield* decodeElfVersions(report)
  expect(value.definitions).toEqual(["libexample.so", "EXAMPLE_1.0"])
  expect(value.needs).toEqual([{ library: "libc.so.6", versions: ["GLIBC_2.34", "GLIBC_2.2.5"] }])
  yield* attestElfVersions("libexample.so", ["EXAMPLE_1.0"], value)
  expect((yield* attestElfVersions("libexample.so", ["EXAMPLE_0.9"], value).pipe(Effect.either))._tag).toBe("Left")
  expect((yield* decodeElfVersions("No version information found in this file.\n")).needs).toEqual([])
})))
test("missing and truncated version reports cannot qualify compatibility", () => Effect.runPromise(Effect.gen(function* () {
  for (const output of ["", "readelf: error", report.replace("2 entries:", "3 entries:"), report.replace("Cnt: 2", "Cnt: 3"),
    report.replace("File: libc.so.6", "File:"), report.replace(" 0x0020: Name: GLIBC_2.2.5 Flags: none Version: 4\n", "")]) {
    expect((yield* decodeElfVersions(output).pipe(Effect.either))._tag).toBe("Left")
  }
})))
