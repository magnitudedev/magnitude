import { spawn, spawnSync } from "node:child_process"
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it } from "vitest"

const source = readFileSync(join(import.meta.dirname, "../../resources/linux/launch.sh"), "utf8")
const hasFlock = spawnSync("/bin/sh", ["-c", "command -v flock"]).status === 0
const roots: string[] = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

const fixture = () => {
  const root = mkdtempSync(join(tmpdir(), "magnitude-launch-"))
  roots.push(root)
  const gate = join(root, "gate"), bin = join(root, "bin"), app = join(root, "magnitude")
  mkdirSync(gate); mkdirSync(bin)
  writeFileSync(app, `#!/bin/sh\necho "started $*"\n`, { mode: 0o755 })
  writeFileSync(join(bin, "notify-send"), `#!/bin/sh\nprintf '%s\\n' "$@" > "${root}/notified"\n`, { mode: 0o755 })
  const script = join(root, "launch")
  writeFileSync(script, source.replace("gate=/var/lib/magnitude-desktop", `gate=${gate}`).replace("/usr/lib/magnitude-desktop/magnitude", app), { mode: 0o755 })
  const run = () => spawnSync("/bin/sh", [script, "--background"], { encoding: "utf8", env: { ...process.env, PATH: `${bin}:${process.env.PATH}` } })
  const notified = () => { try { return readFileSync(join(root, "notified"), "utf8") } catch { return null } }
  return { gate, run, notified }
}

describe.skipIf(process.platform !== "linux" || !hasFlock)("Linux installed launcher", () => {
  it("starts the application with its arguments when the installation is complete", () => {
    const { gate, run, notified } = fixture()
    writeFileSync(join(gate, "installation.lock"), "")
    chmodSync(join(gate, "installation.lock"), 0o444)
    const result = run()
    expect(result.status).toBe(0)
    expect(result.stdout).toBe("started --background\n")
    expect(notified()).toBeNull()
  })
  it("tells a launch without a terminal that an interrupted package change needs a reinstall", () => {
    const { gate, run, notified } = fixture()
    writeFileSync(join(gate, "installation.lock"), "")
    writeFileSync(join(gate, "installing"), "")
    const result = run()
    expect(result.status).toBe(75)
    expect(result.stdout).toBe("")
    expect(result.stderr).toContain("Reinstall the magnitude-desktop package")
    expect(notified()).toContain("Reinstall the magnitude-desktop package")
  })
  it("reports repair guidance instead of a shell error when the gate is missing", () => {
    const { run, notified } = fixture()
    const result = run()
    expect(result.status).toBe(75)
    expect(result.stderr).toContain("needs package-manager repair")
    expect(notified()).toContain("needs package-manager repair")
  })
  it("asks to retry while the package manager holds the installation", async () => {
    const { gate, run, notified } = fixture()
    const lock = join(gate, "installation.lock")
    writeFileSync(lock, "")
    const holder = spawn("flock", ["-x", lock, "sleep", "30"])
    try {
      await new Promise(resolve => setTimeout(resolve, 300))
      const result = run()
      expect(result.status).toBe(75)
      expect(result.stderr).toContain("after the package manager finishes")
      expect(notified()).toContain("after the package manager finishes")
    } finally { holder.kill() }
  })
})
