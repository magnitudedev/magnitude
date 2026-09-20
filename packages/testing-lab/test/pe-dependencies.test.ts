import { Effect, Schema } from "effect"
import { expect, test } from "vitest"
import { decodePeGraph, decodePeImports } from "../src/pe-dependencies"

const json = Schema.encodeSync(Schema.parseJson(Schema.Unknown))
const symbol = { Name: "CreateFileW", ModuleName: "KERNEL32.dll", ImportByOrdinal: false, Ordinal: 0, DelayImport: false }
const imports = { Imports: [{ Name: "KERNEL32.dll", NumberOfEntries: 2, ImportList: [symbol, { ...symbol, DelayImport: true }] }] }
test("PE native inspector preserves ordinary and delayed imports of the same DLL", async () => {
  expect(await Effect.runPromise(decodePeImports(json(imports)))).toEqual([
    { name: "kernel32.dll", linkage: "required" }, { name: "kernel32.dll", linkage: "delay" },
  ])
})

test("PE native inspector rejects truncated tables, foreign symbols and path-like imports", async () => {
  for (const value of [{}, { Imports: [{ ...imports.Imports[0], NumberOfEntries: 3 }] },
    { Imports: [{ ...imports.Imports[0], NumberOfEntries: 1, ImportList: [{ ...symbol, ModuleName: "other.dll" }] }] },
    { Imports: [{ ...imports.Imports[0], Name: "..\\developer.dll" }] }]) {
    expect((await Effect.runPromise(decodePeImports(json(value)).pipe(Effect.either)))._tag).toBe("Left")
  }
})


const resolution = { ModuleName: "KERNEL32.dll", Filepath: "C:\\Windows\\System32\\kernel32.dll", SearchStrategy: 1 }
const root = { Filepath: "C:\\App\\app.exe", Imports: imports.Imports, Dependencies: [resolution] }
const graph = { schemaVersion: 1, Root: root.Filepath, Modules: [root] }
test("PE graph binds native resolutions and import declarations to its root context", async () => {
  const observed = await Effect.runPromise(decodePeGraph(json(graph), "c:\\app\\APP.exe"))
  expect(observed.get("c:\\app\\app.exe")?.Dependencies[0]?.Filepath).toBe("C:\\Windows\\System32\\kernel32.dll")
  expect(observed.size).toBe(1)
})

test("PE graph rejects another root, omitted or duplicate contexts and nonlocal loader paths", async () => {
  for (const value of [{ ...graph, Modules: [] }, { ...graph, Root: "C:\\Other\\app.exe" },
    { ...graph, Modules: [root, root] },
    { ...graph, Modules: [{ ...root, Dependencies: [{ ...resolution, Filepath: "\\\\server\\share\\owned.dll" }] }] }, {}]) {
    expect((await Effect.runPromise(decodePeGraph(json(value), root.Filepath).pipe(Effect.either)))._tag).toBe("Left")
  }
})

test("PE imports include Windows driver modules such as the print spooler", async () => {
  const observed = await Effect.runPromise(decodePeImports(json({ Imports: [{ Name: "WINSPOOL.DRV", NumberOfEntries: 1,
    ImportList: [{ ...symbol, Name: "OpenPrinterW", ModuleName: "WINSPOOL.DRV", DelayImport: true }] }] })))
  expect(observed).toEqual([{ name: "winspool.drv", linkage: "delay" }])
})
