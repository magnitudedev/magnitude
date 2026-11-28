import { chmodSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { describe, expect, it } from "vitest"
import { findOpenCodeDesktopInstallation } from "./opencode"

const touch = (file: string, executable = false) => {
  mkdirSync(dirname(file), { recursive: true })
  writeFileSync(file, "")
  if (executable) chmodSync(file, 0o755)
}

describe("findOpenCodeDesktopInstallation", () => {
  it("detects OpenCode Windows desktop app in LOCALAPPDATA without probing GUI --version", () => {
    const home = mkdtempSync(join(tmpdir(), "magnitude-opencode-win-"))
    const localAppData = join(home, "AppData", "Local")
    const gui = join(localAppData, "OpenCode", "OpenCode.exe")
    touch(gui)

    expect(
      findOpenCodeDesktopInstallation({ platform: "win32", environment: { LOCALAPPDATA: localAppData }, home }),
    ).toEqual({ executable: gui, probeVersion: false })
  })

  it("prefers bundled opencode-cli.exe sidecar over GUI OpenCode.exe on Windows", () => {
    const home = mkdtempSync(join(tmpdir(), "magnitude-opencode-win-sidecar-"))
    const localAppData = join(home, "AppData", "Local")
    const gui = join(localAppData, "OpenCode", "OpenCode.exe")
    const cli = join(localAppData, "OpenCode", "opencode-cli.exe")
    touch(gui)
    touch(cli)

    expect(
      findOpenCodeDesktopInstallation({ platform: "win32", environment: { LOCALAPPDATA: localAppData }, home }),
    ).toEqual({ executable: cli, probeVersion: true })
  })

  it("detects OpenCode Windows desktop app under Programs\\@opencode-aidesktop and ProgramFiles", () => {
    const home = mkdtempSync(join(tmpdir(), "magnitude-opencode-win-programs-"))
    const localAppData = join(home, "AppData", "Local")
    const programFiles = join(home, "Program Files")
    const programFilesGui = join(programFiles, "OpenCode", "OpenCode.exe")
    touch(programFilesGui)

    expect(
      findOpenCodeDesktopInstallation({
        platform: "win32",
        environment: { LOCALAPPDATA: localAppData, ProgramFiles: programFiles },
        home,
      }),
    ).toEqual({ executable: programFilesGui, probeVersion: false })

    const desktopGui = join(localAppData, "Programs", "@opencode-aidesktop", "OpenCode.exe")
    touch(desktopGui)

    expect(
      findOpenCodeDesktopInstallation({
        platform: "win32",
        environment: { LOCALAPPDATA: localAppData, ProgramFiles: programFiles },
        home,
      }),
    ).toEqual({ executable: desktopGui, probeVersion: false })
  })

  it("detects standalone ~/.opencode/bin/opencode.exe on Windows", () => {
    const home = mkdtempSync(join(tmpdir(), "magnitude-opencode-win-bin-"))
    const cli = join(home, ".opencode", "bin", "opencode.exe")
    touch(cli)

    expect(
      findOpenCodeDesktopInstallation({
        platform: "win32",
        environment: { LOCALAPPDATA: join(home, "AppData", "Local") },
        home,
      }),
    ).toEqual({ executable: cli, probeVersion: true })
  })

  it("detects OpenCode macOS app bundle sidecar and GUI fallback in ~/Applications", () => {
    const home = mkdtempSync(join(tmpdir(), "magnitude-opencode-mac-"))
    const gui = join(home, "Applications", "OpenCode.app", "Contents", "MacOS", "OpenCode")
    touch(gui, true)

    expect(
      findOpenCodeDesktopInstallation({ platform: "darwin", environment: {}, home }),
    ).toEqual({ executable: gui, probeVersion: false })

    const cli = join(home, "Applications", "OpenCode.app", "Contents", "MacOS", "opencode-cli")
    touch(cli, true)

    expect(
      findOpenCodeDesktopInstallation({ platform: "darwin", environment: {}, home }),
    ).toEqual({ executable: cli, probeVersion: true })
  })

  it("detects standalone ~/.opencode/bin/opencode on Linux and returns undefined when absent", () => {
    const home = mkdtempSync(join(tmpdir(), "magnitude-opencode-linux-"))
    expect(findOpenCodeDesktopInstallation({ platform: "linux", environment: {}, home })).toBeUndefined()

    const cli = join(home, ".opencode", "bin", "opencode")
    touch(cli, true)
    expect(findOpenCodeDesktopInstallation({ platform: "linux", environment: {}, home })).toEqual({
      executable: cli,
      probeVersion: true,
    })
  })
})
