import { Effect, Either } from "effect"
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { findExecutable, harnessCommand } from "./executable"

const npmGlobalDirectory = () => {
  const directory = mkdtempSync(join(tmpdir(), "magnitude-npm-shims-"))
  for (const file of ["codex", "codex.cmd", "codex.ps1"]) writeFileSync(join(directory, file), "")
  return directory
}

describe("findExecutable", () => {
  it("never resolves npm's extensionless POSIX shim as a Windows executable", () => {
    const directory = npmGlobalDirectory()
    expect(findExecutable("codex", directory, "win32", ".com;.exe;.bat;.cmd")).toBe(join(directory, "codex.cmd"))
  })

  it("prefers earlier PATHEXT extensions and earlier PATH entries on Windows", () => {
    const first = mkdtempSync(join(tmpdir(), "magnitude-path-first-"))
    const second = npmGlobalDirectory()
    writeFileSync(join(second, "codex.exe"), "")
    expect(findExecutable("codex", `${first};${second}`, "win32", ".exe;.cmd")).toBe(join(second, "codex.exe"))
    writeFileSync(join(first, "codex.cmd"), "")
    expect(findExecutable("codex", `${first};${second}`, "win32", ".exe;.cmd")).toBe(join(first, "codex.cmd"))
  })

  it("ignores Windows directories named like an executable", () => {
    const directory = mkdtempSync(join(tmpdir(), "magnitude-path-dir-"))
    mkdirSync(join(directory, "codex.cmd"))
    expect(findExecutable("codex", directory, "win32", ".cmd")).toBeUndefined()
  })
})

describe("harnessCommand", () => {
  const run = (executable: string, args: ReadonlyArray<string>, platform: NodeJS.Platform) =>
    Effect.runSync(Effect.either(harnessCommand(executable, args, platform)))

  it("spawns native executables directly", () => {
    const command = Either.getOrThrow(run("C:\\Tools\\codex.exe", ["debug", "models"], "win32"))
    expect(command).toMatchObject({ command: "C:\\Tools\\codex.exe", args: ["debug", "models"], shell: false })
    expect(Either.getOrThrow(run("/usr/local/bin/codex.cmd", ["--version"], "darwin"))).toMatchObject({ shell: false })
  })

  it("runs Windows batch shims through cmd.exe as one quoted command line", () => {
    const command = Either.getOrThrow(run("C:\\Users\\Q A\\AppData\\Roaming\\npm\\codex.CMD", ["debug", "models", "--bundled", "C:\\a b"], "win32"))
    expect(command).toMatchObject({
      command: "\"C:\\Users\\Q A\\AppData\\Roaming\\npm\\codex.CMD\" debug models --bundled \"C:\\a b\"",
      args: [],
      shell: true,
    })
  })

  it("rejects tokens cmd.exe would reinterpret", () => {
    expect(Either.isLeft(run("C:\\npm\\pi.cmd", ["install", "%PATH%"], "win32"))).toBe(true)
    expect(Either.isLeft(run("C:\\npm\\pi.cmd", ["install", "a\"b"], "win32"))).toBe(true)
  })
})
