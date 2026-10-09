import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const root = fileURLToPath(new URL("../../", import.meta.url))
const sources = (directory: string): string[] => readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
  const path = join(directory, entry.name)
  if (entry.isDirectory()) return entry.name === "node_modules" || entry.name === "out" || entry.name === "dist" ? [] : sources(path)
  return /\.(ts|tsx)$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name) ? [path] : []
})
const files = [...sources(join(root, "web/src")), ...sources(join(root, "desktop/src"))]

describe("dialogs", () => {
  it("never uses the browser's native dialogs", () => {
    expect(files.filter(file => /(^|[^.\w])(window\.)?(confirm|alert|prompt)\(/m.test(readFileSync(file, "utf8")))).toEqual([])
  })
  it("never prompts on unload", () => {
    expect(files.filter(file => /returnValue\s*=|preventDefault\(\)[^\n]*beforeunload|onbeforeunload/.test(readFileSync(file, "utf8")))).toEqual([])
  })
  it("uses Electron's native dialogs only where no window can show the app's own", () => {
    const main = readFileSync(join(root, "desktop/src/main.ts"), "utf8")
    const calls = [...main.matchAll(/dialog\.(show\w+)\(/g)].map(match => match[1])
    expect(calls.sort()).toEqual(["showErrorBox", "showErrorBox", "showErrorBox", "showMessageBox"].sort())
    expect(files.filter(file => !file.endsWith("main.ts") && /\bdialog\.show\w+\(/.test(readFileSync(file, "utf8")))).toEqual([])
  })
})
