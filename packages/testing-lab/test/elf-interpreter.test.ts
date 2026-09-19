import { Effect, Option } from "effect"
import { expect, test } from "vitest"
import { decodeElfInterpreter } from "../src/elf-interpreter"
import { linuxOsDependencies } from "../src/suites/linux-package-dependencies"

test("ELF program headers require the architecture's real system loader", () => Effect.runPromise(Effect.gen(function* () {
  const header = "Elf file type is DYN (Position-Independent Executable file)\nEntry point 0x1000\nProgram Headers:\n"
  const segment = " INTERP 0x0000 0x0000\n [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]\n"
  expect(Option.getOrThrow((yield* decodeElfInterpreter(header + segment, "x64")).path)).toBe("/lib64/ld-linux-x86-64.so.2")
  expect(Option.isNone((yield* decodeElfInterpreter(header + " LOAD 0x1000\n", "x64")).path)).toBe(true)
  for (const output of ["", header + segment + segment, header + " INTERP 0x0000\n", header + segment.replace("/lib64/ld-linux-x86-64.so.2", "/developer/ld.so")]) {
    expect((yield* decodeElfInterpreter(output, "x64").pipe(Effect.either))._tag).toBe("Left")
  }
  expect((yield* decodeElfInterpreter(header + segment, "arm64").pipe(Effect.either))._tag).toBe("Left")
  expect(Option.getOrThrow((yield* decodeElfInterpreter(header + segment.replace("/lib64/ld-linux-x86-64.so.2", "/lib/ld-linux-aarch64.so.1"), "arm64")).path)).toBe("/lib/ld-linux-aarch64.so.1")
})))

test("OS dependency policy cannot supply missing inference or CUDA toolkit libraries", () => {
  const names: readonly string[] = linuxOsDependencies
  for (const name of ["libggml.so", "libggml-base.so", "libcudart.so.12", "libcublas.so.12", "libcuda.so.1"]) expect(names).not.toContain(name)
  expect(names).toContain("libc.so.6")
})
