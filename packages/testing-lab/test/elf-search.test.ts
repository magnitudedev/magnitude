import { Effect } from "effect"
import { expect, test } from "vitest"
import { elfSearch } from "../src/elf-search"

const report = (rpaths: string[], runpaths: string[]) => ({ format: "elf" as const, imports: [], rpaths, runpaths })
test("ELF RPATH reaches children while RUNPATH is restricted to direct imports", () => Effect.runPromise(Effect.gen(function* () {
  const parent = yield* elfSearch("/package", "/package/bin/app", report(["$ORIGIN/../runtime"], []), [])
  expect(parent.direct).toEqual(["/package/runtime"])
  const child = yield* elfSearch("/package", "/package/modules/first.so", report([], ["${ORIGIN}/private"]), parent.inherited)
  expect(child.direct).toEqual(["/package/runtime", "/package/modules/private"])
  expect(child.inherited).toEqual(["/package/runtime"])
  const grandchild = yield* elfSearch("/package", "/package/modules/private/second.so", report([], []), child.inherited)
  expect(grandchild.direct).toEqual(["/package/runtime"])
  const modern = yield* elfSearch("/package", "/package/bin/app", report(["$ORIGIN/old"], ["$ORIGIN/new"]), [])
  expect(modern.direct).toEqual(["/package/bin/new"])
  expect(modern.inherited).toEqual([])
})))

test("ELF search rejects cwd, ambient libraries, unsupported tokens and root escapes", () => Effect.runPromise(Effect.gen(function* () {
  for (const path of ["", ".", "/usr/local/lib", "/opt/toolkit/lib", "$ORIGIN/../../outside", "$ORIGIN/$LIB", "${ORIGIN}/$PLATFORM", "$ORIGIN//absolute"]) {
    expect((yield* elfSearch("/package", "/package/bin/app", report([], [path]), []).pipe(Effect.either))._tag).toBe("Left")
  }
  expect((yield* elfSearch("/package", "/package/bin/app", report(["/developer/lib"], ["$ORIGIN"]), []).pipe(Effect.either))._tag).toBe("Left")
  expect((yield* elfSearch("/package", "/package/bin/app", report([], []), ["/elsewhere"]).pipe(Effect.either))._tag).toBe("Left")
})))
