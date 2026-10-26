import { NodeContext } from "@effect/platform-node"
import { Effect } from "effect"
import { spawnSync } from "node:child_process"
import { generateKeyPairSync } from "node:crypto"
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { renderUnixInstallationScript } from "./installation-scripts"

const template = readFileSync(new URL("../../resources/install.sh", import.meta.url), "utf8")
const helpers = template.slice(template.indexOf("stop_progress() {"), template.indexOf("# Answers come from the terminal"))
const header = `set -eu\nprogress_pid=''\nprogress_terminal=no\n${helpers}\n`
const shell = (body: string, env: Record<string, string> = {}) => spawnSync("/bin/sh", ["-c", header + body], {
  encoding: "utf8", timeout: 15000, env: { ...process.env, ...env },
})

describe.skipIf(process.platform === "win32")("Unix installer progress", () => {
  it("opens the installed Mac app only in a graphical session, with a nonfatal launch fallback", () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-open-"))
    try {
      const bin = join(root, "bin"), resources = join(root, "Magnitude.app/Contents/Resources")
      mkdirSync(bin); mkdirSync(resources, { recursive: true })
      writeFileSync(join(bin, "id"), '#!/bin/sh\necho 1000\n', { mode: 0o700 })
      writeFileSync(join(bin, "launchctl"), '#!/bin/sh\nexit "${TEST_NO_GUI:-0}"\n', { mode: 0o700 })
      writeFileSync(join(resources, "magnitude"), '#!/bin/sh\nprintf "%s\n" "$*" > "$TEST_LAUNCH"\nexit "${TEST_LAUNCH_FAILS:-0}"\n', { mode: 0o700 })
      symlinkSync(join(resources, "magnitude"), join(bin, "magnitude"))
      const open = template.slice(template.indexOf("open_desktop() {"), template.indexOf('\ncase "$(uname -s)" in')).replace("/bin/launchctl", "launchctl")
      const body = `scratch=$TEST_SCRATCH\ndestination=$TEST_DESTINATION\n${open}\nopen_desktop macos`
      const environment = { PATH: `${bin}:${process.env.PATH}`, TEST_SCRATCH: root, TEST_DESTINATION: join(root, "Magnitude.app"), TEST_LAUNCH: join(root, "launched"), SSH_CONNECTION: "", SSH_CLIENT: "", SSH_TTY: "" }
      const skipped = shell(body, { ...environment, TEST_NO_GUI: "1" })
      expect(skipped.status, skipped.stderr).toBe(0)
      expect(skipped.stderr).not.toContain("Opening Magnitude")
      const opened = shell(body, environment)
      expect(opened.status, opened.stderr).toBe(0)
      expect(readFileSync(join(root, "launched"), "utf8").trim()).toBe("app open")
      expect(opened.stderr).toContain("Opened Magnitude")
      const failed = shell(body, { ...environment, TEST_LAUNCH_FAILS: "1" })
      expect(failed.status, failed.stderr).toBe(0)
      expect(failed.stderr).toContain("Open Magnitude from Applications.")
      const linux = body.replace("open_desktop macos", "open_desktop linux")
      const desktop = shell(linux, { ...environment, DISPLAY: ":0" })
      expect(desktop.status, desktop.stderr).toBe(0)
      expect(desktop.stderr).toContain("Opened Magnitude")
      rmSync(join(root, "launched"))
      expect(shell(linux, { ...environment, DISPLAY: "", WAYLAND_DISPLAY: "" }).status).toBe(0)
      expect(shell(linux, { ...environment, DISPLAY: ":0", SSH_CONNECTION: "fixture" }).status).toBe(0)
      expect(existsSync(join(root, "launched"))).toBe(false)
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
  it("shows actual byte progress, average speed and remaining time on both Unix platforms", () => {
    const result = shell(`render_progress 'Downloading app' 12 120000000 200000000 download`)
    expect(result.status, result.stderr).toBe(0)
    expect(result.stdout).toContain("60%  120.0/200.0 MB  10.0 MB/s  ~8s left")
    expect(result.stdout).toContain("[=======-----]")
  })
  it("keeps ETA on narrow terminals without a wrapping progress bar", () => {
    const result = shell(`render_progress 'Downloading app' 12 120000000 200000000 download`, { COLUMNS: "60" })
    expect(result.stdout).not.toContain("[")
    expect(result.stdout).toContain("~8s left")
  })
  it("does not invent estimates before data arrives, or percentages for other steps", () => {
    const zero = shell(`render_progress 'Downloading app' 0 0 200000000 download`)
    expect(zero.stdout).toContain("estimating time remaining")
    expect(zero.stdout).not.toMatch(/NaN|inf/i)
    const stage = shell(`render_progress 'Verifying app signature' 7 0 0 ''`)
    expect(stage.status, stage.stderr).toBe(0)
    expect(stage.stdout).toContain("Verifying app signature · 7s")
    expect(stage.stdout).not.toContain("%")
  })
  it("never sends animation or status into stdout and preserves command failure output", () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-progress-"))
    try {
      const result = shell(`scratch=$TEST_SCRATCH\ntrap stop_progress EXIT\nrun_step 'Installing app' 'Installed app' /bin/sh -c 'printf "native failure\\n" >&2; exit 37'`, { TEST_SCRATCH: root })
      expect(result.status).toBe(37)
      expect(result.stdout).toBe("")
      expect(result.stderr).toContain("Installing app...")
      expect(result.stderr).toContain("native failure")
      expect(result.stderr).not.toContain("✓ Installed app")
      expect(result.stderr).not.toContain("\u001b")
      expect(result.stderr).not.toContain("\r")
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
  it("finishes the indicator before printing server addresses and keys", () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-progress-"))
    try {
      const result = shell(`scratch=$TEST_SCRATCH\ntrap stop_progress EXIT\nrun_step 'Setting up server' 'Server is ready' /bin/sh -c 'printf "Network access key: fixture-key\\n"'`, { TEST_SCRATCH: root })
      expect(result.status, result.stderr).toBe(0)
      expect(result.stdout).toBe("Network access key: fixture-key\n")
      expect(result.stderr).toContain("✓ Server is ready")
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
  it("reports a slow download in a terminal and stops its watcher after completion", async () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-download-progress-"))
    try {
      const bin = join(root, "bin")
      mkdirSync(bin)
      writeFileSync(join(bin, "curl"), `#!/bin/sh\nwhile [ "$#" -gt 0 ]; do case "$1" in --output) output=$2; shift 2;; *) shift;; esac; done\ndd if=/dev/zero of="$output" bs=1000000 count=1 2>/dev/null\nsleep 2\ndd if=/dev/zero bs=1000000 count=1 2>/dev/null >> "$output"\n`, { mode: 0o700 })
      const result = shell(`scratch=$TEST_SCRATCH\nprogress_terminal=yes\ntrap stop_progress EXIT\ndownload https://example.com/app "$scratch/app" 2000000 'Downloading app' 'Downloaded app'\n[ -z "$progress_pid" ]`, {
        PATH: `${bin}:${process.env.PATH}`, TEST_SCRATCH: root,
      })
      expect(result.status, result.stderr).toBe(0)
      expect(result.stdout).toBe("")
      expect(result.stderr).toContain("50%  1.0/2.0 MB")
      expect(result.stderr).toContain("\r  [")
      expect(result.stderr).not.toContain("\r\u001b[K  [")
      expect(result.stderr).toContain("MB/s")
      expect(result.stderr).toContain("s left")
      expect(result.stderr).toContain("✓ Downloaded app · 2.0 MB")
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
  it("cleans up the progress process and returns 130 on terminal Ctrl-C", async () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-progress-cancel-"))
    try {
      const publicKey = generateKeyPairSync("ed25519").publicKey.export({ type: "spki", format: "pem" }).toString()
      const script = await Effect.runPromise(renderUnixInstallationScript({ origin: "https://magnitude.dev", appleTeam: "ABCDEFGHIJ", publicKey }).pipe(Effect.provide(NodeContext.layer)))
      const setup = script.slice(0, script.indexOf('\ncase "$(uname -s)" in'))
      const fixture = join(root, "cancel.sh")
      writeFileSync(fixture, `${setup}\nstart_progress 'Verifying app signature'\nprintf '%s' "$progress_pid" > "$TEST_PROGRESS_PID"\nwhile :; do sleep 1; done\n`)
      const python = `import os,pty,select,signal,sys,time\npid,fd=pty.fork()\nif pid==0:\n signal.signal(signal.SIGINT,signal.SIG_DFL)\n os.execv('/bin/sh',['sh',sys.argv[1]])\noutput=b''\nwhile b'Verifying app signature' not in output:\n ready,_,_=select.select([fd],[],[],5)\n if not ready: raise RuntimeError('no installer progress')\n output+=os.read(fd,4096)\nwhile not os.path.exists(os.environ['TEST_PROGRESS_PID']): time.sleep(0.01)\nos.write(fd,b'\\x03')\nwhile True:\n try:\n  chunk=os.read(fd,4096)\n  if not chunk: break\n  output+=chunk\n except OSError: break\n_,status=os.waitpid(pid,0)\nprint(os.waitstatus_to_exitcode(status))\n`
      const result = spawnSync("python3", ["-c", python, fixture], { encoding: "utf8", timeout: 10000, env: { ...process.env, TEST_PROGRESS_PID: join(root, "pid") } })
      expect(result.status, result.stderr).toBe(0)
      expect(result.stdout.trim()).toBe("130")
      const progressPid = Number(readFileSync(join(root, "pid"), "utf8"))
      expect(() => process.kill(progressPid, 0)).toThrow()
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
})

const powershell = process.env.MAGNITUDE_TEST_POWERSHELL ?? (process.platform === "win32" ? "powershell.exe" : "pwsh")
const hasPowerShell = spawnSync(powershell, ["-NoLogo", "-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"], { encoding: "utf8" }).status === 0

describe.skipIf(!hasPowerShell)("Windows installer progress", () => {
  it("parses the complete script and exercises download display and native process handling", () => {
    const result = spawnSync(powershell, ["-NoLogo", "-NoProfile", "-NonInteractive", "-File", fileURLToPath(new URL("./fixtures/installation-progress.ps1", import.meta.url)), fileURLToPath(new URL("../../resources/install.ps1", import.meta.url))], { encoding: "utf8", timeout: 20000 })
    expect(result.status, result.stdout + result.stderr).toBe(0)
    expect(result.stdout).toContain("Windows progress checks passed")
    expect(result.stderr).toContain("60%  120.0/200.0 MB  10.0 MB/s  ~8s left")
    expect(result.stderr).toContain("child output")
    expect(result.stderr).not.toMatch(/NaN|Infinity/)
    expect(result.stderr).not.toContain("\u001b")
  })
})
